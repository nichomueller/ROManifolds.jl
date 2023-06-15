function reduce_fe_operator(info,feop,solver,rbspace,s)
  ϵ = info.ϵ
  # fun_mdeim = info.fun_mdeim
  rb_res = compress_residual(feop,solver,rbspace,s;ϵ)
  rb_jac = compress_jacobian(feop,solver,rbspace,s;ϵ)
  RBOperator(rb_jac,rb_res;st_mdeim)
end

function reduce_fe_operator(info,feop,solver,rbspace,s)
  ϵ = info.ϵ
  st_mdeim = info.st_mdeim
  # fun_mdeim = info.fun_mdeim
  res_ad = compress_residual(feop,solver,rbspace,s;ϵ,st_mdeim)
  jac_ad = compress_jacobian(feop,solver,rbspace,s;ϵ,st_mdeim)
  RBOperator(res_ad,jac_ad;st_mdeim)
end

struct RBOperator
  res::Function
  jac::Function

  function RBOperator(res_ad,jac_ad;kwargs...)
    res = collect_residual_contributions(res_ad;kwargs...)
    jac = collect_jacobian_contributions(jac_ad;kwargs...)
    new(res,jac)
  end
end

function collect_residual_contributions(res_ad::LazyArray;kwargs...)
  map(collect_residual_contributions,res_ad)
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

    function get_residual_coefficient(res_ad::$Tad;st_mdeim=true)

    end
  end

end
