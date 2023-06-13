abstract type Affinity end
struct ParamAffinity <: Affinity end
struct TimeAffinity <: Affinity end
struct ParamTimeAffinity <: Affinity end

_affinity(args...) = nothing
_affinity(aff::ParamAffinity,args...) = aff
_affinity(::Nothing,aff::TimeAffinity) = aff
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
      p_aff = nothing
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

  params_test = rand(params,n_tests)
  times_test = rand(params,n_tests)

  p_aff = ParamAffinity()
  for μ = params_test
    t = first(times_test)
    ratio = d(μ,t) ./ didx(μ,t)
    if !all(ratio .== ratio[1])
      p_aff = nothing
      break
    end
  end

  t_aff = TimeAffinity()
  for t = times_test
    μ = first(params_test)
    ratio = d(μ,t) ./ didx(μ,t)
    if !all(ratio .== ratio[1])
      t_aff = nothing
      break
    end
  end

  _affinity(p_aff,t_aff)
end
