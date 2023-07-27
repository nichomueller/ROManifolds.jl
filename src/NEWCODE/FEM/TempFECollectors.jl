################################################################################
abstract type CollectorMap <: Map end

return_cache(::CollectorMap,x...) = nothing

evaluate!(cache,k::CollectorMap,x...) = k.f(x...)

return_value(k::CollectorMap,x...) = k.f(x...)

struct CollectSolutionsMap <: CollectorMap
  f::Function

  function CollectSolutionsMap(
    feop::ParamTransientFEOperator,
    fesolver::ODESolver)

    new(μ -> solve(feop,fesolver,μ,fesolver.uh0(μ)))
  end
end

function return_cache(k::CollectSolutionsMap,params::Table)
  μ = testitem(params)
  vμ = return_value(k,μ)
  cμ = return_cache(vμ)
  cμ
end

function evaluate!(cache,k::CollectSolutionsMap,params::Table)
  pmap(eachindex(params)) do i
    vμ = return_value(k,params[i])
    evaluate!(cache,vμ)
    println(norm(cache))
    cache
  end
end

try
catch
  k = CollectSolutionsMap(feop,fesolver)
  sols,pparams = load_test(info,feop,fesolver,10)
  c = return_cache(k,pparams)
  sols = evaluate!(c,k,pparams)
end

################################################################################
A = [rand(100,100) for _ = 1:10]
Clazy = lazy_map(svd,A)
cache = array_cache(C)
for i in eachindex(C)
  getindex!(cache,C,i)
end

C = map(svd,A)

_rand_gen_matrix(::Int) = rand(100,100)
_rand_gen_matrix(n::Vector{Int}) = lazy_map(_rand_gen_matrix,n)

my_lazy_fun1(n::Vector{Int}) = hcat(_rand_gen_matrix(n)...)
@time case1 = my_lazy_fun1([1,2,3,4,5])

# myop = Operation(hcat)
evaluate(hcat,_rand_gen_matrix([1,2,3,4,5])...)
