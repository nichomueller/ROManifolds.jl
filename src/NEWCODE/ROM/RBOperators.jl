function reduce_fe_operator(
  info::RBInfo,
  feop::ParamTransientFEOperator{Top},
  fesolver::ODESolver) where Top

  ϵ = info.ϵ
  # fun_mdeim = info.fun_mdeim
  nsnaps = info.nsnaps
  params = realization(feop,nsnaps)
  sols = generate_solutions(feop,fesolver,params)
  rbspace = compress_solution(sols,feop,fesolver;ϵ)

  nsnaps = info.nsnaps_mdeim
  #compress_residual_and_jacobian(...)
  rb_res = compress_residual(feop,fesolver,rbspace,sols,params;ϵ,nsnaps)
  rb_jac = compress_jacobian(feop,fesolver,rbspace,sols,params;ϵ,nsnaps)
  rbop = RBOperator{Top}(rb_jac,rb_res,rbspace;st_mdeim)
  save(info,rbop)

  return rbop
end

struct RBOperator{Top<:OperatorType}
  res::Function
  jac::Function
  rbspace::Any
end

for Tad in (:RBAffineDecomposition,:TransientRBAffineDecomposition)

  @eval begin
    function collect_residual_contributions(res_ad::Vector{$Tad};kwargs...)
      (u,μ) -> sum([collect_residual_contributions(r;kwargs...)(u,μ)
        for r in res_ad])
    end

    function collect_residual_contributions(res_ad::$Tad;kwargs...)
      res_basis = get_basis(res_ad)
      nrows = get_nrows(res_ad)
      ncols = get_ncols(res_ad)
      res_rb = zeros(nrows,ncols)
      function res_rb!(u,μ)
        coeff = get_residual_coefficient(res_ad;kwargs...)(u,μ)
        r = @distributed (+) for (b,c) in zip(res_basis,coeff)
          LinearAlgebra.kron(b,c)
        end
        copyto!(res_rb,r)
      end
      res_rb!
    end

    function collect_jacobian_contributions(jac_ad::Vector{$Tad};kwargs...)
      (u,μ) -> sum([collect_jacobian_contributions(j;kwargs...)(u,μ)
        for j in jac_ad])
    end

    function collect_jacobian_contributions(jac_ad::$Tad;kwargs...)
      jac_basis = get_basis(jac_ad)
      nrows = get_nrows(jac_ad)
      ncols = get_ncols(jac_ad)
      jac_rb = zeros(nrows,ncols)
      function jac_rb!(u,μ)
        coeff = get_jacobian_coefficient(jac_ad;kwargs...)(u,μ)
        j = @distributed (+) for (b,c) in zip(jac_basis,coeff)
          LinearAlgebra.kron(b,c)
        end
        copyto!(jac_rb,j)
      end
      jac_rb!
    end

  end

end

function get_residual_coefficient(
  res_ad::TransientRBAffineDecomposition;
  st_mdeim=true)

end

function solve_mdeim(mdeim_interp::LU,a::AbstractArray)
  Pa = mdeim_interp.P*a
  y = mdeim_interp.L \ Pa
  x = mdeim_interp.U \ y
  x'
end

function recast_coefficient(
  solver::ODESolver,
  basis_time::Tuple{Vararg{AbstractMatrix}},
  coeff::AbstractMatrix)

  time_ndofs = get_time_ndofs(solver)
  bt,btbt,btbt_shift = basis_time
  Qs = size(coeff,2)
  Qt = size(basis_time,2)
  sorted_idx(qs) = [(i-1)*Qs+qs for i = 1:Qt]

  rcoeff = allocate_matrix(coeff,time_ndofs,size(coeff,2))
  @inbounds for q = 1:Qs
    rcoeff[:,q] = bt*coeff[sorted_idx(qs)]
  end

  rcoeff
end

function project_residual_coefficient(
  ::ODESolver,
  basis_time::Tuple{Vararg{AbstractMatrix}},
  coeff::AbstractMatrix)

  bt,btbt,btbt_shift = basis_time
  proj = Matrix{Float}[]
  @inbounds for q = axes(coeff,2), ijt = axes(bt,2)
    push!(proj,sum(bt[:,ijt].*coeff[:,q]))
  end
  proj
end

function project_jacobian_coefficient(
  ::θMethod,
  basis_time::NTuple{3,AbstractMatrix},
  coeff::AbstractMatrix)

  bt,btbt,btbt_shift = basis_time
  proj = Matrix{Float}[]
  proj_shift = Matrix{Float}[]
  @inbounds for q = axes(coeff,2), ijt = axes(btbt,2)
    push!(proj,sum(btbt[:,ijt].*coeff[:,q]))
    push!(proj_shift,sum(btbt_shift[:,ijt].*coeff[2:Nt,q]))
  end
  proj + proj_shift
end
