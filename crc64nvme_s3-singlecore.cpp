#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace crc64nvme {

constexpr std::uint64_t kInit = 0xFFFFFFFFFFFFFFFFull;
constexpr std::uint64_t kXorOut = 0xFFFFFFFFFFFFFFFFull;
// Reflected form of 0xAD93D23594C93659, because CRC-64/NVME uses refin=true/refout=true.
constexpr std::uint64_t kReflectedPoly = 0x9A6C9329AC4BC9B5ull;

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

    void update(const std::vector<char>& buffer, std::size_t len) {
        update(reinterpret_cast<const std::uint8_t*>(buffer.data()), len);
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
    std::vector<char> buffer(1024 * 1024);

    while (in) {
        in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = in.gcount();
        if (count > 0) {
            hasher.update(buffer, static_cast<std::size_t>(count));
        }
    }

    if (!in.eof() && in.fail()) {
        throw std::runtime_error("Failed while reading input stream");
    }

    return hasher.digest();
}

bool self_test() {
    const std::string sample = "123456789";
    Hasher h;
    h.update(reinterpret_cast<const std::uint8_t*>(sample.data()), sample.size());
    const auto crc = h.digest();
    const auto b64 = s3_base64(crc);
    const bool ok = (crc == 0xAE8B14860A799888ull) && (b64 == "rosUhgp5mIg=");

    std::cout << "self-test input : " << sample << '\n';
    std::cout << "expected hex    : ae8b14860a799888\n";
    std::cout << "actual hex      : " << to_hex(crc) << '\n';
    std::cout << "expected base64 : rosUhgp5mIg=\n";
    std::cout << "actual base64   : " << b64 << '\n';
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
            std::ifstream file(argv[1], std::ios::binary);
            if (!file) {
                std::cerr << "Could not open file: " << argv[1] << '\n';
                return 2;
            }
            crc = crc64nvme::hash_stream(file);
        } else {
            std::cerr << "Usage:\n"
                      << "  " << argv[0] << " [file]\n"
                      << "  " << argv[0] << " --self-test\n";
            return 2;
        }

        std::cout << "CRC64NVME hex:    " << crc64nvme::to_hex(crc) << '\n';
        std::cout << "S3 header value:  " << crc64nvme::s3_base64(crc) << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
