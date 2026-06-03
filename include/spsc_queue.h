#ifndef GRAPH_ALGORITHM_SPSC_QUEUE_H
#define GRAPH_ALGORITHM_SPSC_QUEUE_H

#include <array>
#include <atomic>
#include <cstddef>

// Single-producer single-consumer bounded queue.
// Capacity is the number of usable elements; one extra slot is kept internally
// to distinguish full from empty.
template <typename T, std::size_t Capacity>
class SpscQueue {
public:
    static_assert(Capacity > 0, "SpscQueue capacity must be positive");

    bool try_push(const T& value) {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        const std::size_t next_tail = increment(tail);
        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false;
        }
        buffer_[tail] = value;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    bool try_pop(T& value) {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        if (head == tail_.load(std::memory_order_acquire)) {
            return false;
        }
        value = buffer_[head];
        head_.store(increment(head), std::memory_order_release);
        return true;
    }

    bool peek(T& value) const {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        if (head == tail_.load(std::memory_order_acquire)) {
            return false;
        }
        value = buffer_[head];
        return true;
    }

    bool empty() const {
        return head_.load(std::memory_order_acquire) ==
               tail_.load(std::memory_order_acquire);
    }

private:
    static constexpr std::size_t kStorageSize = Capacity + 1;

    static std::size_t increment(std::size_t index) {
        return (index + 1) % kStorageSize;
    }

    std::array<T, kStorageSize> buffer_{};
    std::atomic<std::size_t> head_{0};
    std::atomic<std::size_t> tail_{0};
};

#endif  // GRAPH_ALGORITHM_SPSC_QUEUE_H
