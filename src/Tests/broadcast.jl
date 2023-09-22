op = feop
ode_op = get_algebraic_operator(op)
K = 2
μ = realization(op,K)
nfree = num_free_dofs(test)
w = get_free_dof_values(uh0μ(μ))
wf = copy(w)
w0 = copy(w)

w0 .= pi
wf .= w0
@. wf = wf .* 8 + w0 .* 2

struct PTBroadcasted{T}
  array::PTArray{T}
end

function Base.broadcasted(f,a::PTBroadcasted,b::PTBroadcasted)
  PTBroadcasted(map(f,a.array,b.array))
end

function Base.broadcasted(f,a::Number,b::PTArray)
  PTBroadcasted(PTArray(map(x->f(a,x),b.array)))
end

function Base.broadcasted(f,a::PTArray,b::Number)
  PTBroadcasted(PTArray(map(x->f(x,b),a.array)))
end

function Base.broadcasted(
  f,a::Union{PTArray,PTBroadcasted},
  b::Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{0}})
  Base.broadcasted(f,a,Base.materialize(b))
end

function Base.broadcasted(
  f,a::Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{0}},
  b::Union{PTArray,PTBroadcasted})
  Base.broadcasted(f,Base.materialize(a),b)
end

function Base.materialize(b::PTBroadcasted{T}) where T
  a = similar(b.array)
  Base.materialize!(a,b)
  a
end

function Base.materialize!(a::PTArray,b::Broadcast.Broadcasted)
  map(x->Base.materialize!(x,b),a.array)
  a
end

function Base.materialize!(a::PTArray,b::PTBroadcasted)
  map(Base.materialize!,a.array,b.array.array)
end
