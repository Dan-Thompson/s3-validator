#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace crc64nvme {

constexpr std::uint64_t kInit = 0xFFFFFFFFFFFFFFFFull;
constexpr std::uint64_t kXorOut = 0xFFFFFFFFFFFFFFFFull;
constexpr std::uint64_t kReflectedPoly = 0x9A6C9329AC4BC9B5ull;

constexpr std::size_t kReadBufferSize = 1ull << 20; // 1 MiB
constexpr std::size_t kMinChunkSize   = 4ull << 20; // 4 MiB
constexpr std::size_t kMaxChunkSize   = 64ull << 20; // 64 MiB

std::array<std::uint64_t, 256> make_table() {
    std::array<std::uint64_t, 256> table{};
    for (std::uint64_t i = 0; i < 256; ++i) {
        std::uint64_t crc = i;
        for (int bit = 0; bit < 8; ++bit) {
            if (crc & 1ull) {
                crc = (crc >> 1) ^ kReflectedPoly;
            } else {
                crc >>= 1;
            }
        }
        table[static_cast<std::size_t>(i)] = crc;
    }
    return table;
}

const std::array<std::uint64_t, 256> kTable = make_table();

class Hasher {
public:
    Hasher() : crc_(kInit) {}

    void update(const std::uint8_t* data, std::size_t len) {
        for (std::size_t i = 0; i < len; ++i) {
            const std::uint8_t idx = static_cast<std::uint8_t>(crc_ ^ data[i]);
            crc_ = kTable[idx] ^ (crc_ >> 8);
        }
    }

    void update(const char* data, std::size_t len) {
        update(reinterpret_cast<const std::uint8_t*>(data), len);
    }

    std::uint64_t digest() const {
        return crc_ ^ kXorOut;
    }

private:
    std::uint64_t crc_;
};

std::string to_hex(std::uint64_t value) {
    std::ostringstream oss;
    oss << std::hex << std::nouppercase << std::setfill('0') << std::setw(16) << value;
    return oss.str();
}

std::array<std::uint8_t, 8> to_big_endian_bytes(std::uint64_t value) {
    return {
        static_cast<std::uint8_t>((value >> 56) & 0xFF),
        static_cast<std::uint8_t>((value >> 48) & 0xFF),
        static_cast<std::uint8_t>((value >> 40) & 0xFF),
        static_cast<std::uint8_t>((value >> 32) & 0xFF),
        static_cast<std::uint8_t>((value >> 24) & 0xFF),
        static_cast<std::uint8_t>((value >> 16) & 0xFF),
        static_cast<std::uint8_t>((value >> 8) & 0xFF),
        static_cast<std::uint8_t>(value & 0xFF)
    };
}

std::string base64_encode(const std::uint8_t* data, std::size_t len) {
    static constexpr char kAlphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string out;
    out.reserve(((len + 2) / 3) * 4);

    for (std::size_t i = 0; i < len; i += 3) {
        const std::uint32_t b0 = data[i];
        const std::uint32_t b1 = (i + 1 < len) ? data[i + 1] : 0;
        const std::uint32_t b2 = (i + 2 < len) ? data[i + 2] : 0;
        const std::uint32_t triple = (b0 << 16) | (b1 << 8) | b2;

        out.push_back(kAlphabet[(triple >> 18) & 0x3F]);
        out.push_back(kAlphabet[(triple >> 12) & 0x3F]);
        out.push_back((i + 1 < len) ? kAlphabet[(triple >> 6) & 0x3F] : '=');
        out.push_back((i + 2 < len) ? kAlphabet[triple & 0x3F] : '=');
    }

    return out;
}

std::string s3_base64(std::uint64_t crc) {
    const auto bytes = to_big_endian_bytes(crc);
    return base64_encode(bytes.data(), bytes.size());
}

std::uint64_t hash_stream(std::istream& in) {
    Hasher hasher;
    std::vector<char> buffer(kReadBufferSize);

    while (in) {
        in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = in.gcount();
        if (count > 0) {
            hasher.update(buffer.data(), static_cast<std::size_t>(count));
        }
    }

    if (!in.eof() && in.fail()) {
        throw std::runtime_error("Failed while reading input stream");
    }

    return hasher.digest();
}

std::size_t pick_chunk_size(std::uint64_t file_size, unsigned threads) {
    if (threads == 0) {
        threads = 1;
    }
    const std::uint64_t target_chunks = static_cast<std::uint64_t>(threads) * 4ull;
    const std::uint64_t raw = (file_size + target_chunks - 1) / target_chunks;
    const std::uint64_t clamped = std::min<std::uint64_t>(
        kMaxChunkSize,
        std::max<std::uint64_t>(kMinChunkSize, raw)
    );
    return static_cast<std::size_t>(clamped);
}

std::uint64_t gf2_matrix_times(const std::array<std::uint64_t, 64>& mat,
                               std::uint64_t vec) {
    std::uint64_t sum = 0;
    std::size_t idx = 0;
    while (vec != 0) {
        if (vec & 1ull) {
            sum ^= mat[idx];
        }
        vec >>= 1;
        ++idx;
    }
    return sum;
}

std::array<std::uint64_t, 64> gf2_matrix_square(
    const std::array<std::uint64_t, 64>& mat) {
    std::array<std::uint64_t, 64> square{};
    for (std::size_t n = 0; n < 64; ++n) {
        square[n] = gf2_matrix_times(mat, mat[n]);
    }
    return square;
}

// Combine two finalized CRC64/NVME values:
// crc(A || B) = combine(crc(A), crc(B), len(B))
std::uint64_t combine(std::uint64_t crc1, std::uint64_t crc2, std::uint64_t len2) {
    if (len2 == 0) {
        return crc1;
    }

    std::array<std::uint64_t, 64> odd{};
    std::array<std::uint64_t, 64> even{};

    odd[0] = kReflectedPoly;
    std::uint64_t row = 1;
    for (std::size_t n = 1; n < 64; ++n) {
        odd[n] = row;
        row <<= 1;
    }

    even = gf2_matrix_square(odd);
    odd = gf2_matrix_square(even);

    do {
        even = gf2_matrix_square(odd);
        if (len2 & 1ull) {
            crc1 = gf2_matrix_times(even, crc1);
        }
        len2 >>= 1;
        if (len2 == 0) {
            break;
        }

        odd = gf2_matrix_square(even);
        if (len2 & 1ull) {
            crc1 = gf2_matrix_times(odd, crc1);
        }
        len2 >>= 1;
    } while (len2 != 0);

    return crc1 ^ crc2;
}

std::uint64_t hash_file_range(std::ifstream& in,
                              std::vector<char>& buffer,
                              std::uint64_t offset,
                              std::uint64_t length) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error("File offset is too large for this platform");
    }

    in.clear();
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!in) {
        throw std::runtime_error("Failed to seek in input file");
    }

    Hasher hasher;
    std::uint64_t remaining = length;

    while (remaining > 0) {
        const std::size_t want = static_cast<std::size_t>(
            std::min<std::uint64_t>(remaining, buffer.size())
        );
        in.read(buffer.data(), static_cast<std::streamsize>(want));
        const std::streamsize got = in.gcount();
        if (got <= 0) {
            throw std::runtime_error("Unexpected end of file while reading input file");
        }
        hasher.update(buffer.data(), static_cast<std::size_t>(got));
        remaining -= static_cast<std::uint64_t>(got);
    }

    return hasher.digest();
}

std::uint64_t hash_file_parallel(const std::string& path) {
    const std::uint64_t file_size =
        static_cast<std::uint64_t>(std::filesystem::file_size(path));

    if (file_size == 0) {
        return 0;
    }

    unsigned workers = std::thread::hardware_concurrency();
    if (workers == 0) {
        workers = 1;
    }

    const std::size_t chunk_size = pick_chunk_size(file_size, workers);
    const std::size_t chunk_count =
        static_cast<std::size_t>((file_size + chunk_size - 1) / chunk_size);

    workers = std::min<unsigned>(workers, static_cast<unsigned>(chunk_count));

    std::vector<std::uint64_t> chunk_crcs(chunk_count, 0);
    std::atomic<std::size_t> next_chunk{0};

    std::exception_ptr first_error;
    std::mutex error_mutex;

    auto worker = [&]() {
        try {
            std::ifstream in(path, std::ios::binary);
            if (!in) {
                throw std::runtime_error("Could not open file: " + path);
            }

            std::vector<char> buffer(kReadBufferSize);

            while (true) {
                const std::size_t index =
                    next_chunk.fetch_add(1, std::memory_order_relaxed);
                if (index >= chunk_count) {
                    return;
                }

                const std::uint64_t offset =
                    static_cast<std::uint64_t>(index) * chunk_size;
                const std::uint64_t length =
                    std::min<std::uint64_t>(chunk_size, file_size - offset);

                chunk_crcs[index] = hash_file_range(in, buffer, offset, length);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!first_error) {
                first_error = std::current_exception();
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (unsigned i = 0; i < workers; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    if (first_error) {
        std::rethrow_exception(first_error);
    }

    std::uint64_t crc = chunk_crcs[0];
    for (std::size_t i = 1; i < chunk_count; ++i) {
        const std::uint64_t length = (i + 1 == chunk_count)
            ? (file_size - static_cast<std::uint64_t>(i) * chunk_size)
            : static_cast<std::uint64_t>(chunk_size);

        crc = combine(crc, chunk_crcs[i], length);
    }

    return crc;
}

bool self_test() {
    const std::string sample = "123456789";

    Hasher whole_hasher;
    whole_hasher.update(sample.data(), sample.size());
    const std::uint64_t whole = whole_hasher.digest();

    Hasher a;
    a.update(sample.data(), 4);
    const std::uint64_t crc_a = a.digest();

    Hasher b;
    b.update(sample.data() + 4, sample.size() - 4);
    const std::uint64_t crc_b = b.digest();

    const std::uint64_t combined = combine(crc_a, crc_b, sample.size() - 4);

    const bool ok =
        (whole == 0xAE8B14860A799888ull) &&
        (combined == whole) &&
        (s3_base64(whole) == "rosUhgp5mIg=");

    std::cout << "self-test input : " << sample << '\n';
    std::cout << "expected hex    : ae8b14860a799888\n";
    std::cout << "actual hex      : " << to_hex(whole) << '\n';
    std::cout << "combined hex    : " << to_hex(combined) << '\n';
    std::cout << "expected base64 : rosUhgp5mIg=\n";
    std::cout << "actual base64   : " << s3_base64(whole) << '\n';
    std::cout << (ok ? "SELF-TEST PASSED\n" : "SELF-TEST FAILED\n");

    return ok;
}

} // namespace crc64nvme

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            return crc64nvme::self_test() ? 0 : 1;
        }

        std::uint64_t crc = 0;

        if (argc == 1) {
#ifdef _WIN32
            _setmode(_fileno(stdin), _O_BINARY);
#endif
            crc = crc64nvme::hash_stream(std::cin);
        } else if (argc == 2) {
            crc = crc64nvme::hash_file_parallel(argv[1]);
        } else {
            std::cerr << "Usage:\n"
                      << "  " << argv[0] << " [file]\n"
                      << "  " << argv[0] << " --self-test\n";
            return 2;
        }

        //std::cout << "CRC64NVME hex:    " << crc64nvme::to_hex(crc) << '\n';
        //std::cout << "S3 header value:  " << crc64nvme::s3_base64(crc) << '\n';
        std::cout << crc64nvme::s3_base64(crc) << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
