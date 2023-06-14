abstract type Affinity end
struct ParamAffinity <: Affinity end
struct TimeAffinity <: Affinity end
struct ParamTimeAffinity <: Affinity end
struct NonAffinity <: Affinity end

_affinity(args...) = NonAffinity()
_affinity(aff::ParamAffinity,args...) = aff
_affinity(::NonAffinity,aff::TimeAffinity) = aff
_affinity(::ParamAffinity,::TimeAffinity) = ParamTimeAffinity()

function Affinity(data,params::Table;n_tests=10)
  d(μ) = first(first(data(μ)))
  global idx
  for (i,ci) in enumerate(d(rand(params)))
    if !isapprox(sum(abs.(ci)),0.)
      idx = i
      break
    end
  end
  didx(μ) = max.(d(μ)[idx],eps())

  params_test = rand(params,n_tests)

  p_aff = ParamAffinity()
  for μ = params_test
    ratio = d(μ) ./ didx(μ)
    if !all(ratio .== ratio[1])
      p_aff = NonAffinity()
      break
    end
  end

  _affinity(p_aff)
end

function Affinity(data,params::Table,times::AbstractVector;n_tests=10)
  d(μ,t) = first(first(data(μ,t)))
  global idx
  for (i,ci) in enumerate(d(rand(params),rand(times)))
    if !isapprox(sum(abs.(ci)),0.)
      idx = i
      break
    end
  end
  didx(μ,t) = max.(d(μ,t)[idx],eps())

  param_base = rand(params)
  time_base = rand(times)
  datum_idx_base = didx(param_base,time_base)

  params_test = rand(params,n_tests)
  times_test = rand(times,n_tests)

  p_aff = ParamAffinity()
  for μ = params_test
    t = first(times_test)
    ratio = didx(μ,t) ./ datum_idx_base
    if !all(ratio .== ratio[1])
      p_aff = NonAffinity()
      break
    end
  end

  t_aff = TimeAffinity()
  for t = times_test
    μ = first(params_test)
    ratio = didx(μ,t) ./ datum_idx_base
    if !all(ratio .== ratio[1])
      t_aff = NonAffinity()
      break
    end
  end

  _affinity(p_aff,t_aff)
end
