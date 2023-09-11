struct PTFunction <: Function
  f::Function
  params::AbstractArray
  times::Union{Real,Vector{<:Real}}
end

const PTField = Union{GenericField{PTFunction},
                      FieldGradient{N,GenericField{PTFunction}} where N}
const GenericPTField = Union{PTField,ZeroField{<:PTField}}

function get_params(a::GenericField{PTFunction})
  a.object.params
end

function get_times(a::GenericField{PTFunction})
  a.object.times
end

function get_field(a::GenericField{PTFunction},p,t)
  GenericField(a.object.f(p,t))
end

function get_params(a::FieldGradient{N,GenericField{PTFunction}} where N)
  get_params(a.object)
end
function get_times(a::FieldGradient{N,GenericField{PTFunction}} where N)
  get_times(a.object)
end

function get_field(a::FieldGradient{N,GenericField{PTFunction}} where N,p,t)
  FieldGradient{N}(get_field(a.object,p,t))
end

function get_nfields(a::PTField)
  length(get_params(a))*length(get_times(a))
end

function Arrays.testitem(a::PTField)
  p,t = get_params(a),get_times(a)
  p1,t1 = map(testitem,(p,t))
  f = get_field(a,p1,t1)
  return f
end

function Arrays.return_cache(fpt::GenericPTField,x::AbstractArray{<:Point})
  n = get_nfields(fpt)
  f = testitem(fpt)
  cb,cf = return_cache(f,x)
  ca = PTArray(fill(cb.array,n))
  ca,cb,cf
end

function Arrays.evaluate!(cache,fpt::GenericPTField,x::AbstractArray{<:Point})
  ca,c... = cache
  p,t = get_params(fpt),get_times(fpt)
  nt = length(t)
  @inbounds for q = eachindex(ca)
    pq = p[fast_idx(q,nt)]
    tq = t[slow_idx(q,nt)]
    fq = get_field(fpt,pq,tq)
    ca[q] = evaluate!(c,fq,x)
  end
  ca
end

op,solver = feop,fesolver
a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
v = get_fe_basis(test)
u = get_trial_fe_basis(allocate_trial_space(trial))
params = realization(op,2)
times = get_times(solver)
a(params[1],times[1])*∇(v)⋅∇(u)
pf = PTFunction(a,params,times)
pf*∇(v)⋅∇(u)
boh
