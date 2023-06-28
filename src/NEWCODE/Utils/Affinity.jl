abstract type Affinity end
struct ParamAffinity <: Affinity end
struct TimeAffinity <: Affinity end
struct ParamTimeAffinity <: Affinity end
struct NonAffinity <: Affinity end

_affinity(args...) = NonAffinity()
_affinity(::ParamAffinity,args...) = ParamAffinity()
_affinity(::NonAffinity,::TimeAffinity) = TimeAffinity()
_affinity(::ParamAffinity,::TimeAffinity) = ParamTimeAffinity()

function get_affinity(::FESolver,params::Table,data::Function;ntests=10)
  d(μ) = first(first(last(data(μ))))
  global idx
  for (i,ci) in enumerate(d(rand(params)))
    if !isapprox(sum(abs.(ci)),0.)
      idx = i
      break
    end
  end
  didx(μ) = max.(abs.(d(μ)[idx]),eps())

  params_test = rand(params,ntests)

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

function get_affinity(solver::ODESolver,params::Table,data::Function;ntests=10)
  times = get_times(solver)
  d(μ,t) = first(first(last(data(μ,t))))
  global idx
  for (i,ci) in enumerate(d(rand(params),rand(times)))
    if !isapprox(sum(abs.(ci)),0.)
      idx = i
      break
    end
  end
  didx(μ,t) = max.(abs.(d(μ,t)[idx]),eps())

  param_base = rand(params)
  time_base = rand(times)
  datum_idx_base = didx(param_base,time_base)

  params_test = rand(params,ntests)
  times_test = rand(times,ntests)

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

function Gridap.CellData.get_data(
  ::ParamAffinity,
  ::FESolver,
  params::Table,
  data::Function)

  μ = rand(params)
  [data(μ)]
end

function Gridap.CellData.get_data(
  ::NonAffinity,
  ::FESolver,
  params::Table,
  data::Function)

  pmap(μ->data(μ),params)
end

function Gridap.CellData.get_data(
  ::ParamTimeAffinity,
  solver::ODESolver,
  params::Table,
  data::Function)

  times = get_times(solver)
  μ = rand(params)
  t = rand(times)
  [data(μ,t)]
end

function Gridap.CellData.get_data(
  ::ParamAffinity,
  solver::ODESolver,
  params::Table,
  data::Function)

  times = get_times(solver)
  μ = rand(params)
  pmap(t->data(μ,t),times)
end

function Gridap.CellData.get_data(
  ::TimeAffinity,
  solver::ODESolver,
  params::Table,
  data::Function)

  times = get_times(solver)
  t = rand(times)
  pmap(μ->data(μ,t),params)
end

function Gridap.CellData.get_data(
  ::NonAffinity,
  solver::ODESolver,
  params::Table,
  data::Function)

  times = get_times(solver)
  d = map(params) do μ
    vcat(map(t->data(μ,t),times)...)
  end
  vcat(d...)
end
