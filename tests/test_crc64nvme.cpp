#define CRC64NVME_NO_MAIN
#include "../crc64nvme_s3-multicore.cpp"

#include <memory>
#include <random>

namespace {

// Deliberately independent of the production tables and accelerated kernels.
std::uint64_t reference_crc(const std::uint8_t* data, std::size_t length) {
    std::uint64_t crc = ~std::uint64_t{0};
    for (std::size_t i = 0; i < length; ++i) {
        crc ^= data[i];
        for (unsigned bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ ((crc & 1) ? 0x9A6C9329AC4BC9B5ull : 0);
        }
    }
    return ~crc;
}

void require(bool passed, const char* label, std::size_t length = 0,
             std::size_t offset = 0) {
    if (!passed) {
        throw std::runtime_error(std::string(label) + ": length=" +
                                 std::to_string(length) + ", offset=" +
                                 std::to_string(offset));
    }
}

void check_hash(const std::uint8_t* data, std::size_t length, std::size_t offset,
                std::mt19937_64& random) {
    const auto expected = reference_crc(data + offset, length);
    crc64nvme::Hasher whole, incremental;
    whole.update(data + offset, length);
    require(whole.digest() == expected, "whole buffer", length, offset);

    std::size_t position = 0;
    while (position < length) {
        incremental.update(data + offset + position, 0);
        const auto count = std::min<std::size_t>(length - position,
                                                1 + random() % 2001);
        incremental.update(data + offset + position, count);
        position += count;
    }
    require(incremental.digest() == expected, "incremental", length, offset);
}

void check_large_combinations(std::mt19937_64& random) {
    for (unsigned sample = 0; sample < 16; ++sample) {
        const auto crc = random();
        auto one_byte = crc;
        for (unsigned bit = 0; bit < 8; ++bit) {
            one_byte = (one_byte >> 1) ^
                       ((one_byte & 1) ? 0x9A6C9329AC4BC9B5ull : 0);
        }
        require(crc64nvme::combine(crc, 0, 1) == one_byte, "one-byte shift");

        // Exercise every cached exponent, including the top bit of uint64_t,
        // without allocating the corresponding (potentially enormous) input.
        for (unsigned bit = 1; bit < 64; ++bit) {
            const auto half = std::uint64_t{1} << (bit - 1);
            const auto doubled = crc64nvme::combine(
                crc64nvme::combine(crc, 0, half), 0, half);
            require(crc64nvme::combine(crc, 0, half << 1) == doubled,
                    "doubled shift", bit);
        }
        const auto left = random() >> 1;
        const auto right = random() >> 1;
        require(crc64nvme::combine(crc64nvme::combine(crc, 0, left), 0, right) ==
                crc64nvme::combine(crc, 0, left + right), "shift composition");
    }
}

#ifdef __linux__
void check_linux_io(const std::vector<std::uint8_t>& data) {
    std::uint8_t byte;
    bool rejected = false;
    try {
        crc64nvme::read_some(-1, &byte, 1);
    } catch (const std::system_error&) {
        rejected = true;
    }
    require(rejected, "invalid input descriptor");

    std::unique_ptr<std::FILE, decltype(&std::fclose)> stream(std::tmpfile(),
                                                           &std::fclose);
    require(stream != nullptr, "temporary truncated input");
    // Fill and reuse the ring, then test worker cleanup with a size that runs
    // past EOF after several complete chunks and a partial final chunk.
    for (unsigned i = 0; i < 2; ++i) {
        require(std::fwrite(data.data(), 1, data.size(), stream.get()) == data.size(),
                "write truncated input");
    }
    std::vector<std::uint8_t> contents(data);
    contents.insert(contents.end(), data.begin(), data.end());
    const auto expected = reference_crc(contents.data(), contents.size());
    std::rewind(stream.get());
    require(crc64nvme::hash_file_pipeline(fileno(stream.get()), contents.size(), 2) == expected,
            "ordered pipeline buffers");
    std::rewind(stream.get());
    require(crc64nvme::hash_file_ranges(fileno(stream.get()), contents.size(), 3) == expected,
            "parallel file ranges");
    std::rewind(stream.get());
    rejected = false;
    try {
        crc64nvme::hash_file_pipeline(fileno(stream.get()), 12ull << 20, 2);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "truncated pipeline input");
    rejected = false;
    try {
        crc64nvme::hash_file_ranges(fileno(stream.get()), 12ull << 20, 3);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "truncated parallel input");
}
#endif

} // namespace

int main() {
    try {
        std::mt19937_64 random(0x4e564d45);
        std::vector<std::uint8_t> data((2 << 20) + 64);
        for (auto& byte : data) {
            byte = static_cast<std::uint8_t>(random());
        }

        const std::string sample = "123456789";
        require(reference_crc(reinterpret_cast<const std::uint8_t*>(sample.data()),
                              sample.size()) == 0xAE8B14860A799888ull,
                "reference check value");
        require(crc64nvme::s3_base64(0xAE8B14860A799888ull) == "rosUhgp5mIg=",
                "S3 byte order and base64");
        crc64nvme::Hasher empty;
        empty.update(data.data(), 0);
        require(empty.digest() == 0, "empty input");
        require(crc64nvme::combine(0, 0, 0) == 0, "empty combination");

        for (std::size_t offset = 0; offset < 32; ++offset) {
            for (std::size_t length = 0; length < 320; ++length) {
                check_hash(data.data(), length, offset, random);
            }
        }
        const std::size_t lengths[] = {
            511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049,
            4095, 4096, 4097, 8191, 8192, 8193, 16383, 16384, 16385,
            65535, 65536, 65537, (1 << 20) - 1, 1 << 20, (1 << 20) + 1,
            (2 << 20) + 13
        };
        for (auto length : lengths) {
            for (std::size_t offset : {0u, 1u, 7u, 15u}) {
                check_hash(data.data(), length, offset, random);
            }
        }
        for (unsigned sample_index = 0; sample_index < 300; ++sample_index) {
            const std::size_t length = random() % 65538;
            const std::size_t split = random() % (length + 1);
            const std::size_t offset = random() % 32;
            crc64nvme::Hasher left, right;
            left.update(data.data() + offset, split);
            right.update(data.data() + offset + split, length - split);
            require(crc64nvme::combine(left.digest(), right.digest(), length - split) ==
                    reference_crc(data.data() + offset, length),
                    "combined buffers", length, split);
        }
        check_large_combinations(random);
#ifdef __linux__
        check_linux_io(data);
#endif

        std::unique_ptr<std::FILE, decltype(&std::fclose)> stream(std::tmpfile(),
                                                               &std::fclose);
        require(stream != nullptr, "temporary input stream");
        require(std::fwrite(sample.data(), 1, sample.size(), stream.get()) == sample.size(),
                "write input stream");
        std::rewind(stream.get());
        require(crc64nvme::hash_stream(stream.get()) == 0xAE8B14860A799888ull, "stream");
        require(crc64nvme::hash_stream(stream.get()) == 0, "stream at EOF");

        std::cout << "CRC64/NVME regression tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failed: " << error.what() << '\n';
        return 1;
    }
}
