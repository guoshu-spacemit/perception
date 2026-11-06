#ifndef OPENCL_PROFILING_H
#define OPENCL_PROFILING_H

#include <CL/cl.h>
#include <limits>

namespace las2::detail
{

// Profiling is optional telemetry, not a condition for using completed GPU
// output. Keep unavailable samples distinct from a measured zero duration.
template <typename Query> double opencl_event_ms(cl_event event, Query query)
{
    const double unavailable = std::numeric_limits<double>::quiet_NaN();
    if (!event)
        return unavailable;

    cl_ulong start = 0;
    cl_ulong end = 0;
    if (query(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr) != CL_SUCCESS ||
        query(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr) != CL_SUCCESS ||
        end < start)
        return unavailable;

    // Subtract integers before conversion to preserve precision when the
    // device's absolute timestamp is large; validate before unsigned subtraction.
    return static_cast<double>(end - start) / 1.0e6;
}

inline double opencl_event_ms(cl_event event)
{
    return opencl_event_ms(event, clGetEventProfilingInfo);
}

} // namespace las2::detail

#endif // OPENCL_PROFILING_H
