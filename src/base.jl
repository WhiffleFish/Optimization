function optimize end

function optimize_info end

function optimize(opt, args...; kwargs...)
    f, x, hist = optimize_info(opt, args...; kwargs...)
    return f, x
end
