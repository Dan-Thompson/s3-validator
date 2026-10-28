#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#if !defined(_WIN32) && !defined(_FILE_OFFSET_BITS)
#define _FILE_OFFSET_BITS 64
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef __linux__
#include <cerrno>
#include <condition_variable>
#include <fcntl.h>
#include <linux/magic.h>
#include <sched.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/sysmacros.h>
#include <system_error>
#include <unistd.h>
#endif
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <sys/types.h>
#endif

#if !defined(CRC64NVME_PORTABLE) && \
    defined(__aarch64__) && \
    (defined(__APPLE__) || defined(__ARM_FEATURE_CRYPTO) || \
     (defined(__linux__) && (defined(__GNUC__) || defined(__clang__)))) && \
    (!defined(__BYTE_ORDER__) || __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#include <arm_neon.h>
#define CRC64NVME_ARM_PMULL 1
#if defined(__linux__) && !defined(__ARM_FEATURE_CRYPTO)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#define CRC64NVME_ARM_RUNTIME 1
#define CRC64NVME_HW_TARGET __attribute__((target("+crypto")))
#else
#define CRC64NVME_HW_TARGET
#endif
#elif !defined(CRC64NVME_PORTABLE) && \
      (defined(__x86_64__) || defined(__i386__)) && \
      (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define CRC64NVME_X86_PCLMUL 1
#define CRC64NVME_HW_TARGET __attribute__((target("pclmul,sse2")))
#endif

namespace crc64nvme {

constexpr std::uint64_t kInit = 0xFFFFFFFFFFFFFFFFull;
constexpr std::uint64_t kXorOut = 0xFFFFFFFFFFFFFFFFull;
constexpr std::uint64_t kReflectedPoly = 0x9A6C9329AC4BC9B5ull;

constexpr std::size_t kReadBufferSize = 1ull << 20; // 1 MiB
constexpr std::size_t kMinChunkSize   = 4ull << 20; // 4 MiB

// Each slice advances a byte by one more position in the CRC. Keeping these
// tables in read-only memory avoids startup work and processes 16 bytes at once.
constexpr auto make_tables() {
    std::array<std::array<std::uint64_t, 256>, 16> tables{};
    for (std::uint64_t i = 0; i < 256; ++i) {
        std::uint64_t crc = i;
        for (int bit = 0; bit < 8; ++bit) {
            if (crc & 1ull) {
                crc = (crc >> 1) ^ kReflectedPoly;
            } else {
                crc >>= 1;
            }
        }
        tables[0][static_cast<std::size_t>(i)] = crc;
    }
    for (std::size_t slice = 1; slice < tables.size(); ++slice) {
        for (std::size_t i = 0; i < 256; ++i) {
            const auto crc = tables[slice - 1][i];
            tables[slice][i] = tables[0][crc & 0xff] ^ (crc >> 8);
        }
    }
    return tables;
}

alignas(64) constexpr auto kTables = make_tables();

#if defined(CRC64NVME_ARM_PMULL) || defined(CRC64NVME_X86_PCLMUL)

// bit_reverse(x^n mod polynomial), represented in reflected CRC bit order.
// Generate every coefficient at compile time; no runtime polynomial division.
constexpr std::uint64_t fold_constant(unsigned n) {
    std::uint64_t value = std::uint64_t{1} << 63;
    for (unsigned i = 0; i < n; ++i) {
        value = (value >> 1) ^ ((value & 1) ? kReflectedPoly : 0);
    }
    return value;
}

constexpr auto make_fold_coefficients() {
    std::array<std::array<std::uint64_t, 2>, 8> coefficients{};
    for (unsigned i = 0; i < coefficients.size(); ++i) {
        const unsigned bits = (i + 1) * 128;
        coefficients[i] = {fold_constant(bits + 63), fold_constant(bits - 1)};
    }
    return coefficients;
}
constexpr auto kFoldCoefficients = make_fold_coefficients();

#ifdef CRC64NVME_ARM_PMULL
using CrcVector = uint64x2_t;
CRC64NVME_HW_TARGET inline CrcVector vector_load(const std::uint8_t* p) {
    return vreinterpretq_u64_u8(vld1q_u8(p));
}
CRC64NVME_HW_TARGET inline CrcVector vector_xor(CrcVector a, CrcVector b) {
    return veorq_u64(a, b);
}
CRC64NVME_HW_TARGET inline CrcVector vector_crc(std::uint64_t crc) {
    return vsetq_lane_u64(crc, vdupq_n_u64(0), 0);
}
CRC64NVME_HW_TARGET inline CrcVector vector_fold(CrcVector x, CrcVector coefficients) {
    const auto low = vmull_p64(vgetq_lane_u64(x, 0), vgetq_lane_u64(coefficients, 0));
    const auto high = vmull_p64(vgetq_lane_u64(x, 1), vgetq_lane_u64(coefficients, 1));
    return veorq_u64(vreinterpretq_u64_p128(low), vreinterpretq_u64_p128(high));
}
CRC64NVME_HW_TARGET inline void vector_store(std::uint8_t* p, CrcVector x) {
    vst1q_u8(p, vreinterpretq_u8_u64(x));
}
#else
using CrcVector = __m128i;
CRC64NVME_HW_TARGET inline CrcVector vector_load(const std::uint8_t* p) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
}
CRC64NVME_HW_TARGET inline CrcVector vector_xor(CrcVector a, CrcVector b) {
    return _mm_xor_si128(a, b);
}
CRC64NVME_HW_TARGET inline CrcVector vector_crc(std::uint64_t crc) {
    return _mm_set_epi64x(0, static_cast<long long>(crc));
}
CRC64NVME_HW_TARGET inline CrcVector vector_fold(CrcVector x, CrcVector coefficients) {
    return _mm_xor_si128(_mm_clmulepi64_si128(x, coefficients, 0x00),
                         _mm_clmulepi64_si128(x, coefficients, 0x11));
}
CRC64NVME_HW_TARGET inline void vector_store(std::uint8_t* p, CrcVector x) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), x);
}
#endif

// Eight independent polynomial accumulators hide multiplication latency.
// Consumes only complete 128-byte blocks. The caller hashes the remaining tail.
CRC64NVME_HW_TARGET inline std::uint64_t update_hardware_blocks(
    std::uint64_t crc, const std::uint8_t*& data, std::size_t& len) {
    const auto* cursor = data;
    auto remaining = len;
    CrcVector lanes[8];
    for (unsigned i = 0; i < 8; ++i) {
        lanes[i] = vector_load(cursor + i * 16);
    }
    lanes[0] = vector_xor(lanes[0], vector_crc(crc));
    cursor += 128;
    remaining -= 128;
    const auto stride = vector_load(reinterpret_cast<const std::uint8_t*>(kFoldCoefficients[7].data()));
    while (remaining >= 128) {
        for (unsigned i = 0; i < 8; ++i) {
            lanes[i] = vector_xor(vector_fold(lanes[i], stride), vector_load(cursor + i * 16));
        }
        cursor += 128;
        remaining -= 128;
    }
    auto folded = lanes[7];
    for (unsigned i = 0; i < 7; ++i) {
        const auto coefficients = vector_load(reinterpret_cast<const std::uint8_t*>(kFoldCoefficients[6 - i].data()));
        folded = vector_xor(folded, vector_fold(lanes[i], coefficients));
    }
    // The folded block represents the same polynomial as the entire input.
    // Reduce its 16 bytes through the portable tables, avoiding a second reducer.
    alignas(16) std::uint8_t bytes[16];
    vector_store(bytes, folded);
    crc = 0;
    for (unsigned i = 0; i < 16; ++i) {
        crc ^= kTables[15 - i][bytes[i]];
    }
    data = cursor;
    len = remaining;
    return crc;
}

inline bool hardware_available() {
#if defined(CRC64NVME_ARM_RUNTIME)
    static const bool supported = (getauxval(AT_HWCAP) & HWCAP_PMULL) != 0;
    return supported;
#elif defined(CRC64NVME_X86_PCLMUL)
    static const bool supported = __builtin_cpu_supports("pclmul") && __builtin_cpu_supports("sse2");
    return supported;
#else
    return true;
#endif
}

inline std::uint64_t update_hardware(
    std::uint64_t crc, const std::uint8_t*& data, std::size_t& len) {
    if (len >= 128 && hardware_available()) {
        return update_hardware_blocks(crc, data, len);
    }
    return crc;
}
#else
inline bool hardware_available() { return false; }
#endif

std::uint64_t load_le64(const std::uint8_t* data) {
    // memcpy allows unaligned input without violating strict aliasing.
    std::uint64_t value;
    std::memcpy(&value, data, sizeof(value));
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    value = __builtin_bswap64(value);
#endif
    return value;
}

class Hasher {
public:
    Hasher() : crc_(kInit) {}

    void update(const std::uint8_t* data, std::size_t len) {
        auto crc = crc_;
#if defined(CRC64NVME_ARM_PMULL) || defined(CRC64NVME_X86_PCLMUL)
        crc = update_hardware(crc, data, len);
#endif
        while (len >= 16) {
            const auto first = crc ^ load_le64(data);
            crc = kTables[15][ first        & 0xff] ^
                  kTables[14][(first >>  8) & 0xff] ^
                  kTables[13][(first >> 16) & 0xff] ^
                  kTables[12][(first >> 24) & 0xff] ^
                  kTables[11][(first >> 32) & 0xff] ^
                  kTables[10][(first >> 40) & 0xff] ^
                  kTables[ 9][(first >> 48) & 0xff] ^
                  kTables[ 8][ first >> 56] ^
                  kTables[ 7][data[ 8]] ^
                  kTables[ 6][data[ 9]] ^
                  kTables[ 5][data[10]] ^
                  kTables[ 4][data[11]] ^
                  kTables[ 3][data[12]] ^
                  kTables[ 2][data[13]] ^
                  kTables[ 1][data[14]] ^
                  kTables[ 0][data[15]];
            data += 16;
            len -= 16;
        }
        while (len-- != 0) {
            crc = kTables[0][static_cast<std::uint8_t>(crc ^ *data++)] ^ (crc >> 8);
        }
        crc_ = crc;
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

std::uint64_t hash_stream(std::FILE* in) {
    Hasher hasher;
    std::vector<char> buffer(kReadBufferSize);

    // Bulk reads for the buffered FILE interface used on other platforms.
    while (const auto count = std::fread(buffer.data(), 1, buffer.size(), in)) {
        hasher.update(buffer.data(), count);
    }

    if (std::ferror(in)) {
        throw std::runtime_error("Failed while reading input stream");
    }

    return hasher.digest();
}

// Polynomial multiplication in reflected bit order; bit 63 represents one.
constexpr std::uint64_t multiply_mod(std::uint64_t a, std::uint64_t b) {
    std::uint64_t result = 0;
    while (a != 0) {
        if (a & (std::uint64_t{1} << 63)) {
            result ^= b;
        }
        a <<= 1;
        b = (b >> 1) ^ ((b & 1) ? kReflectedPoly : 0);
    }
    return result;
}

constexpr auto make_shift_factors() {
    std::array<std::uint64_t, 64> factors{};
    factors[0] = std::uint64_t{1} << 63;
    for (unsigned bit = 0; bit < 8; ++bit) {
        factors[0] = (factors[0] >> 1) ^ ((factors[0] & 1) ? kReflectedPoly : 0);
    }
    for (std::size_t i = 1; i < factors.size(); ++i) {
        factors[i] = multiply_mod(factors[i - 1], factors[i - 1]);
    }
    return factors;
}

// factors[n] advances a CRC by 2^n bytes. Only 512 bytes of read-only data,
// with no runtime matrix generation or per-range polynomial division.
constexpr auto kShiftFactors = make_shift_factors();

// Combine two finalized CRC64/NVME values:
// crc(A || B) = combine(crc(A), crc(B), len(B))
std::uint64_t combine(std::uint64_t crc1, std::uint64_t crc2, std::uint64_t len2) {
    if (len2 == 0) {
        return crc1;
    }

    std::size_t bit = 0;
    while (len2 != 0) {
        if (len2 & 1ull) {
            crc1 = multiply_mod(kShiftFactors[bit], crc1);
        }
        len2 >>= 1;
        ++bit;
    }

    return crc1 ^ crc2;
}

#ifdef __linux__

constexpr std::size_t kPipelineChunkSize = 1ull << 20; // 1 MiB

class FileDescriptor {
public:
    explicit FileDescriptor(int fd) : fd_(fd) {}
    ~FileDescriptor() { if (fd_ >= 0) ::close(fd_); }
    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;
    int get() const { return fd_; }

private:
    int fd_;
};

unsigned available_cpus() {
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
        const auto count = CPU_COUNT(&allowed);
        if (count > 0) {
            return static_cast<unsigned>(count);
        }
    }
    return std::max(1u, std::thread::hardware_concurrency());
}

std::size_t read_some(int fd, std::uint8_t* data, std::size_t length) {
    for (;;) {
        const auto count = ::read(fd, data, length);
        if (count >= 0) {
            return static_cast<std::size_t>(count);
        }
        if (errno != EINTR) {
            throw std::system_error(errno, std::generic_category(), "Failed to read input");
        }
    }
}

void read_exact(int fd, std::uint8_t* data, std::size_t length) {
    while (length != 0) {
        const auto count = read_some(fd, data, length);
        if (count == 0) {
            throw std::runtime_error("Unexpected end of file while reading input file");
        }
        data += count;
        length -= count;
    }
}

std::uint64_t hash_fd(int fd) {
    Hasher hasher;
    std::unique_ptr<std::uint8_t[]> buffer(new std::uint8_t[kReadBufferSize]);
    while (const auto count = read_some(fd, buffer.get(), kReadBufferSize)) {
        hasher.update(buffer.get(), count);
    }
    return hasher.digest();
}

std::uint64_t hash_file_pipeline(int fd, std::uint64_t file_size, unsigned workers) {
    // One reader preserves sequential HDD access. Workers hash independent
    // chunks while the reader fills the next buffer, also overlapping SSD I/O.
    // Hardware CRC needs few workers; small buffers keep cache handoffs cheap.
    struct Chunk {
        std::unique_ptr<std::uint8_t[]> data;
        std::size_t length = 0;
        std::uint64_t crc = 0;
        bool done = false;
    };
    std::vector<Chunk> slots(workers + 1);
    for (auto& slot : slots) {
        slot.data.reset(new std::uint8_t[kPipelineChunkSize]);
    }

    std::mutex mutex;
    std::condition_variable ready, completed;
    std::uint64_t submitted = 0, claimed = 0, consumed = 0;
    bool stopping = false;
    std::exception_ptr worker_error;

    auto worker = [&] {
        try {
            for (;;) {
                std::unique_lock<std::mutex> lock(mutex);
                ready.wait(lock, [&] { return stopping || claimed < submitted; });
                if (stopping) {
                    return;
                }
                auto& slot = slots[claimed++ % slots.size()];
                lock.unlock();
                Hasher hasher;
                hasher.update(slot.data.get(), slot.length);
                const auto crc = hasher.digest();
                lock.lock();
                slot.crc = crc;
                slot.done = true;
                lock.unlock();
                completed.notify_one();
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                worker_error = std::current_exception();
                stopping = true;
            }
            ready.notify_all();
            completed.notify_one();
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(workers);
    auto stop_and_join = [&] {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        ready.notify_all();
        for (auto& thread : threads) {
            thread.join();
        }
    };

    std::uint64_t crc = 0;
    auto consume = [&] {
        auto& slot = slots[consumed % slots.size()];
        std::unique_lock<std::mutex> lock(mutex);
        completed.wait(lock, [&] { return worker_error || slot.done; });
        if (worker_error) {
            std::rethrow_exception(worker_error);
        }
        lock.unlock();
        crc = consumed == 0 ? slot.crc : combine(crc, slot.crc, slot.length);
        ++consumed;
    };

    try {
        for (unsigned i = 0; i < workers; ++i) {
            threads.emplace_back(worker);
        }
        auto remaining = file_size;
        while (remaining != 0) {
            if (submitted - consumed == slots.size()) {
                consume();
            }
            auto& slot = slots[submitted % slots.size()];
            slot.length = static_cast<std::size_t>(
                std::min<std::uint64_t>(remaining, kPipelineChunkSize));
            slot.done = false;
            read_exact(fd, slot.data.get(), slot.length);
            remaining -= slot.length;
            {
                std::lock_guard<std::mutex> lock(mutex);
                ++submitted;
            }
            ready.notify_one();
        }
        while (consumed != submitted) {
            consume();
        }
    } catch (...) {
        stop_and_join();
        throw;
    }
    stop_and_join();
    return crc;
}

bool prefer_parallel_reads(int fd, const struct stat& info) {
    struct statfs filesystem;
    if (::fstatfs(fd, &filesystem) == 0 &&
        (filesystem.f_type == TMPFS_MAGIC || filesystem.f_type == RAMFS_MAGIC)) {
        return true;
    }
    // Trust the kernel's logical block queue, including device-mapper stacks.
    // Partitions inherit their parent disk's rotational setting. Unknown and
    // network filesystems retain the conservative sequential-reader path.
    std::error_code error;
    auto device = std::filesystem::canonical(
        "/sys/dev/block/" + std::to_string(major(info.st_dev)) + ":" +
        std::to_string(minor(info.st_dev)), error);
    if (error) {
        return false;
    }
    while (device != device.root_path()) {
        std::unique_ptr<std::FILE, decltype(&std::fclose)> rotational(
            std::fopen((device / "queue/rotational").c_str(), "r"), &std::fclose);
        if (rotational) {
            const auto value = std::fgetc(rotational.get());
            if (value == '0' || value == '1') {
                return value == '0';
            }
        }
        device = device.parent_path();
    }
    return false;
}

std::uint64_t hash_file_ranges(int fd, std::uint64_t file_size, unsigned workers) {
    const auto range_size = file_size / workers;
    const auto extra = file_size % workers;
    std::vector<std::uint64_t> crcs(workers);
    std::vector<std::exception_ptr> errors(workers);
    std::atomic<bool> cancelled{false};
    auto worker = [&](unsigned index) {
        try {
            auto offset = index * range_size + std::min<std::uint64_t>(index, extra);
            auto remaining = range_size + (index < extra);
            std::unique_ptr<std::uint8_t[]> buffer(new std::uint8_t[kReadBufferSize]);
            Hasher hasher;
            while (remaining != 0) {
                if (cancelled.load(std::memory_order_relaxed)) {
                    return;
                }
                const auto want = static_cast<std::size_t>(
                    std::min<std::uint64_t>(remaining, kReadBufferSize));
                const auto count = ::pread(fd, buffer.get(), want, static_cast<off_t>(offset));
                if (count < 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    throw std::system_error(errno, std::generic_category(), "Failed to read input");
                }
                if (count == 0) {
                    throw std::runtime_error("Unexpected end of file while reading input file");
                }
                hasher.update(buffer.get(), static_cast<std::size_t>(count));
                offset += static_cast<std::uint64_t>(count);
                remaining -= static_cast<std::uint64_t>(count);
            }
            crcs[index] = hasher.digest();
        } catch (...) {
            errors[index] = std::current_exception();
            cancelled.store(true, std::memory_order_relaxed);
        }
    };
    std::vector<std::thread> threads;
    threads.reserve(workers - 1);
    try {
        for (unsigned i = 1; i < workers; ++i) {
            threads.emplace_back(worker, i);
        }
    } catch (...) {
        cancelled.store(true, std::memory_order_relaxed);
        for (auto& thread : threads) {
            thread.join();
        }
        throw;
    }
    worker(0);
    for (auto& thread : threads) {
        thread.join();
    }
    for (const auto& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
    auto crc = crcs[0];
    for (unsigned i = 1; i < workers; ++i) {
        crc = combine(crc, crcs[i], range_size + (i < extra));
    }
    return crc;
}

std::uint64_t hash_file_parallel(const std::string& path) {
    // Nonblocking open lets fstat reject FIFOs without waiting for a writer.
    // Linux ignores O_NONBLOCK for regular files.
    FileDescriptor file(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK));
    if (file.get() < 0) {
        throw std::system_error(errno, std::generic_category(), "Could not open file: " + path);
    }
    struct stat info;
    if (::fstat(file.get(), &info) != 0) {
        throw std::system_error(errno, std::generic_category(), "Could not stat input file");
    }
    if (!S_ISREG(info.st_mode) || info.st_size < 0) {
        throw std::runtime_error("Input is not a regular file: " + path);
    }
    const auto file_size = static_cast<std::uint64_t>(info.st_size);
    if (file_size == 0) {
        return 0;
    }
    const auto cpus = available_cpus();
    if (file_size > kMinChunkSize && cpus > 1) {
        if (prefer_parallel_reads(file.get(), info)) {
            const auto ranges = 1 + (file_size - 1) / kMinChunkSize;
            const auto workers = static_cast<unsigned>(
                std::min<std::uint64_t>(std::min(cpus, 32u), ranges));
            return hash_file_ranges(file.get(), file_size, workers);
        }
    }
    // Advisory only: unsupported filesystems can ignore this hint.
    (void)posix_fadvise(file.get(), 0, 0, POSIX_FADV_SEQUENTIAL);
    if (file_size > kMinChunkSize && cpus > 1) {
        const auto workers = std::min(cpus - 1, hardware_available() ? 2u : 8u);
        return hash_file_pipeline(file.get(), file_size, workers);
    }

    Hasher hasher;
    const auto buffer_size = static_cast<std::size_t>(
        std::min<std::uint64_t>(kReadBufferSize, file_size));
    std::unique_ptr<std::uint8_t[]> buffer(new std::uint8_t[buffer_size]);
    auto remaining = file_size;
    while (remaining != 0) {
        const auto count = static_cast<std::size_t>(
            std::min<std::uint64_t>(remaining, buffer_size));
        read_exact(file.get(), buffer.get(), count);
        hasher.update(buffer.get(), count);
        remaining -= count;
    }
    return hasher.digest();
}

#else

std::uint64_t hash_file_range(std::FILE* in,
                              std::vector<char>& buffer,
                              std::uint64_t offset,
                              std::uint64_t length,
                              const std::atomic<bool>& cancelled) {
#ifdef _WIN32
    using FileOffset = __int64;
#else
    using FileOffset = off_t;
#endif
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<FileOffset>::max())) {
        throw std::runtime_error("File offset is too large for this platform");
    }

#ifdef _WIN32
    const int seek_result = _fseeki64(in, static_cast<FileOffset>(offset), SEEK_SET);
#else
    const int seek_result = fseeko(in, static_cast<FileOffset>(offset), SEEK_SET);
#endif
    if (seek_result != 0) {
        throw std::runtime_error("Failed to seek in input file");
    }

    Hasher hasher;
    std::uint64_t remaining = length;

    while (remaining > 0) {
        if (cancelled.load(std::memory_order_relaxed)) {
            return 0;
        }
        const std::size_t want = static_cast<std::size_t>(
            std::min<std::uint64_t>(remaining, buffer.size())
        );
        const auto got = std::fread(buffer.data(), 1, want, in);
        if (std::ferror(in) || got != want) {
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

    const auto max_workers = file_size / kMinChunkSize +
                            (file_size % kMinChunkSize != 0);
    workers = static_cast<unsigned>(std::min<std::uint64_t>(workers, max_workers));

    // One contiguous range per worker keeps reads sequential within each file
    // handle and bounds bookkeeping by CPU count, even for very large files.
    const auto range_size = file_size / workers;
    const auto extra = file_size % workers;
    std::vector<std::uint64_t> range_crcs(workers);
    std::atomic<bool> cancelled{false};

    std::exception_ptr first_error;
    std::mutex error_mutex;

    auto worker = [&](unsigned index) {
        try {
            std::unique_ptr<std::FILE, decltype(&std::fclose)> in(
                std::fopen(path.c_str(), "rb"), &std::fclose);
            if (!in) {
                throw std::runtime_error("Could not open file: " + path);
            }

            const auto offset = index * range_size + std::min<std::uint64_t>(index, extra);
            const auto length = range_size + (index < extra);
            std::vector<char> buffer(static_cast<std::size_t>(
                std::min<std::uint64_t>(kReadBufferSize, length)));
            range_crcs[index] = hash_file_range(in.get(), buffer, offset, length, cancelled);
        } catch (...) {
            cancelled.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!first_error) {
                first_error = std::current_exception();
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(workers - 1);
    try {
        for (unsigned i = 1; i < workers; ++i) {
            threads.emplace_back(worker, i);
        }
    } catch (...) {
        cancelled.store(true, std::memory_order_relaxed);
        for (auto& t : threads) {
            t.join();
        }
        throw;
    }
    // The caller is a worker too; small files need no thread creation.
    worker(0);

    for (auto& t : threads) {
        t.join();
    }

    if (first_error) {
        std::rethrow_exception(first_error);
    }

    std::uint64_t crc = range_crcs[0];
    for (unsigned i = 1; i < workers; ++i) {
        crc = combine(crc, range_crcs[i], range_size + (i < extra));
    }

    return crc;
}

#endif

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

#ifndef CRC64NVME_NO_MAIN
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
#ifdef __linux__
            crc = crc64nvme::hash_fd(STDIN_FILENO);
#else
            crc = crc64nvme::hash_stream(stdin);
#endif
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
#endif
