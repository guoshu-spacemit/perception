#ifndef SHARED_INITIALIZATION_CACHE_H
#define SHARED_INITIALIZATION_CACHE_H

#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <utility>

namespace las2::detail
{

// Only live owners retain values. Concurrent requests for one key join its
// initialization; expensive work for different keys never holds the cache lock.
template <class Key, class Value> class SharedInitializationCache
{
public:
    using Pointer = std::shared_ptr<const Value>;

    template <class Factory> Pointer get(const Key& key, Factory&& factory)
    {
        std::shared_ptr<std::promise<Pointer>> producer;
        std::shared_future<Pointer> pending;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto it = entries_.begin(); it != entries_.end();)
            {
                if (!it->second.pending.valid() && it->second.value.expired())
                    it = entries_.erase(it);
                else
                    ++it;
            }
            auto& entry = entries_[key];
            if (auto value = entry.value.lock())
                return value;
            if (!entry.pending.valid())
            {
                producer = std::make_shared<std::promise<Pointer>>();
                entry.pending = producer->get_future().share();
            }
            pending = entry.pending;
        }
        if (!producer)
            return pending.get();

        try
        {
            auto value = std::make_shared<const Value>(std::forward<Factory>(factory)());
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto& entry = entries_.at(key);
                entry.value = value;
                entry.pending = {};
            }
            producer->set_value(value);
            return value;
        }
        catch (...)
        {
            const auto error = std::current_exception();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                entries_.erase(key);
            }
            producer->set_exception(error);
            std::rethrow_exception(error);
        }
    }

private:
    struct Entry
    {
        std::weak_ptr<const Value> value;
        std::shared_future<Pointer> pending;
    };
    std::mutex mutex_;
    std::map<Key, Entry> entries_;
};

} // namespace las2::detail

#endif // SHARED_INITIALIZATION_CACHE_H
