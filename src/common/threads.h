#ifndef KYTY_COMMON_THREADS_H_
#define KYTY_COMMON_THREADS_H_

#include "common/common.h"

#include <cstdint>
#include <memory>
#include <string>

namespace Common {

void InitializeThreads();

// P-core affinity for hybrid CPUs (e.g. i5-14400F: 6 P-cores/12 threads, mask 0x0FFF).
// Pins the calling thread to the P-core mask intersected with the process affinity
// mask and raises/lowers its priority. No-op off Windows. Never fails.
enum class PerfCorePriority { Highest, AboveNormal, BelowNormal };
// Mega-suite fix 5 (P-core topology isolation on i5-14400F, mask 0x0FFF):
// CommandProcessor owns P-Core 0 (0x0003), the Vulkan submit worker owns P-Core 1
// (0x000C), guest JobManager/physics workers spread over P-Cores 2..5 (0x0FF0).
// Splitting the masks keeps the translator and the driver off each other's
// logical cores so they stop evicting each other's L1/L2.
enum class PerfCoreMask : uint32_t {
	AllP = 0x0FFF,
	CommandProcessor = 0x0003,
	SubmitWorker = 0x000C,
	GuestJobs = 0x0FF0,
};
void PinCurrentThreadToPerformanceCores(PerfCorePriority priority = PerfCorePriority::Highest);
void PinCurrentThreadToMask(uint32_t mask, PerfCorePriority priority);

using thread_func_t    = void (*)(void*);
using wait_poll_func_t = void (*)();

struct ThreadPrivate;
struct MutexPrivate;
struct CondVarPrivate;

class Thread {
public:
	Thread(thread_func_t func, void* arg);
	~Thread();

	void Join();
	void Detach();

	// Once a thread has finished, the id may be reused by another thread.
	[[nodiscard]] std::string GetId() const;

	// The id is unique and can't be reused by another thread.
	[[nodiscard]] int GetUniqueId() const;

	static void SleepMicro(uint32_t micros);
	static void SleepNano(uint64_t nanos);
	static bool IsMainThread();

	// Get current thread id
	// Once a thread has finished, the id may be reused by another thread.
	static std::string GetThreadId();

	// Get current thread id
	// The id is unique and can't be reused by another thread.
	static int GetThreadIdUnique();

	KYTY_CLASS_NO_COPY(Thread);

private:
	std::unique_ptr<ThreadPrivate> m_thread;
};

class Mutex {
public:
	Mutex();
	~Mutex();

	void Lock();
	void Unlock();
	bool TryLock();

	friend class CondVar;

	KYTY_CLASS_NO_COPY(Mutex);

private:
	std::unique_ptr<MutexPrivate> m_mutex;
};

class CondVar {
public:
	CondVar();
	~CondVar();

	void Wait(Mutex* mutex);
	bool WaitFor(Mutex* mutex, uint32_t micros);
	void Signal();
	void SignalAll();

	static void SetWaitPollCallback(wait_poll_func_t callback);

	KYTY_CLASS_NO_COPY(CondVar);

private:
	std::unique_ptr<CondVarPrivate> m_cond_var;
};

class LockGuard {
public:
	using mutex_type = Mutex;

	// NOLINTNEXTLINE(google-runtime-references)
	explicit LockGuard(mutex_type& m): m_mutex(m) { m_mutex.Lock(); }

	~LockGuard() { m_mutex.Unlock(); }

	KYTY_CLASS_NO_COPY(LockGuard);

private:
	mutex_type& m_mutex;
};

} // namespace Common

#endif /* KYTY_COMMON_THREADS_H_ */
