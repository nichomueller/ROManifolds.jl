function stretching(x::Point, dim::Int64)
    #=MODIFY
    =#

    m = zeros(length(x))
    m[1] = x[1]^2
    for i in 2:dim
        m[i] = x[i]
    end

    Point(m)

end

struct reference_info <: Int64
    L
    dim
    ndof_dir
end

function generate_cartesian_model(reference_info, deformation)
    #=MODIFY
    =#

    const pmin = Point(Fill(0, reference_info.dim))
    const pmax = Point(Fill(reference_info.L, reference_info.dim))
    const partition = Tuple(Fill(reference_info.ndof_dir, reference_info.dim))

    model = CartesianDiscreteModel(pmin, pmax, partition, map = (x->deformation(x, reference_info.dim)))

    return model

end

