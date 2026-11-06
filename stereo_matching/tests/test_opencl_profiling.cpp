#include "opencl_profiling.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace
{
void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

struct Query
{
    cl_event expected_event;
    cl_ulong start = 1000000;
    cl_ulong end = 3500000;
    cl_int start_status = CL_SUCCESS;
    cl_int end_status = CL_SUCCESS;
    int calls = 0;

    cl_int operator()(
        cl_event event, cl_profiling_info name, size_t size, void* value, size_t* size_ret)
    {
        require(event == expected_event && size == sizeof(cl_ulong) && value && !size_ret,
            "incorrect profiling query arguments");
        require(name == CL_PROFILING_COMMAND_START || name == CL_PROFILING_COMMAND_END,
            "unexpected profiling property");
        ++calls;
        // Deliberately write plausible data even on error: callers must check
        // status, not just timestamps. START failure must short-circuit END.
        const bool is_start = name == CL_PROFILING_COMMAND_START;
        *static_cast<cl_ulong*>(value) = is_start ? start : end;
        return is_start ? start_status : end_status;
    }
};
} // namespace

int main()
{
    try
    {
        int token = 0;
        const auto event = reinterpret_cast<cl_event>(&token); // Never dereferenced.
        Query query{event};
        const auto elapsed = [&]
        {
            query.calls = 0;
            return las2::detail::opencl_event_ms(event, std::ref(query));
        };
        require(elapsed() == 2.5 && query.calls == 2, "valid duration");
        query.end = query.start;
        require(elapsed() == 0.0, "equal timestamps are a valid zero sample");
        query.end = query.start - 1;
        require(std::isnan(elapsed()), "reversed timestamps must not underflow");
        query.end = 0;
        require(std::isnan(elapsed()), "zero end after nonzero start");

        query.start = std::numeric_limits<cl_ulong>::max() - 5000;
        query.end = query.start + 1000;
        require(elapsed() == 0.001, "large absolute timestamps must retain precision");
        query.start_status = CL_PROFILING_INFO_NOT_AVAILABLE;
        require(std::isnan(elapsed()) && query.calls == 1, "failed START query");
        query.start_status = CL_SUCCESS;
        query.end_status = CL_INVALID_EVENT;
        require(std::isnan(elapsed()) && query.calls == 2, "failed END query");
        query.end_status = CL_SUCCESS;
        require(elapsed() == 0.001, "valid sample after error must recover");

        query.calls = 0;
        require(
            std::isnan(las2::detail::opencl_event_ms(nullptr, std::ref(query))) && query.calls == 0,
            "null event must not be queried");
        std::cout << "PASS OpenCL profiling: 9 cases\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
