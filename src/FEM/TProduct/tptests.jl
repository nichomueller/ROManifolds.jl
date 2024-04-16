struct TensorProductMap{D} <: Map
  TensorProductMap(::Val{D}) where D = new{D}()
end

function Arrays.return_cache(
  ::TensorProductMap{D},
  a::Union{AbstractVector{T},NTuple{D,T}}
  ) where {D,T<:AbstractArray}

  ka = kronecker(a...) |> collect
  return ka
end

function Arrays.evaluate!(
  cache,
  ::TensorProductMap{D},
  a::Union{AbstractVector{T},NTuple{D,T}}
  ) where {D,T<:AbstractArray}

  ka = kronecker(a...)
  copyto!(cache,ka)
  return ka
end

struct Temp{D,V,F} <: Field
  a::NTuple{D,Fields.LinearCombinationField{V,F}}
end

for T in (:(Point),:(AbstractVector{<:Point}))
  @eval begin

  function Arrays.return_cache(a::Temp{D},x::$T) where D
    cf,ck = return_cache(a.a[1],x)
    Ck = Vector{typeof(cf)}(undef,D)
    r = return_cache(TensorProductMap(D),Ck)
    return cf,ck,Ck,r
  end

  function Arrays.evaluate!(cache,a::Temp{D},x::$T) where D
    cf,ck,Ck,r = cache
    @inbounds for i = 1:D
      Ck[i] = evaluate!((cf,ck),a.a[i],x)
    end
    evaluate!(r,TensorProductMap(D),Ck)
  end

  end
end

for op in (:(Fields.∇),:(Fields.∇∇))
  @eval begin
    function $op(a::Temp{D}) where D
      aop = ntuple(d->$op(a.a[d]),D)
      Temp(aop)
    end
  end
end

struct TempVec{D,V,F} <: AbstractVector{Temp{V,F}}
  a::NTuple{D,Fields.LinearCombinationFieldVector{V,F}}
end

Base.size(a::TempVec) = (prod(map(x->size(x,1),a.a)),)
Base.getindex(a::TempVec{D},i::Integer) where D = Temp(ntuple(d->getindex(a.a,i),D))
Base.IndexStyle(::Type{<:TempVec}) = IndexLinear()

for T in (:(Point),:(AbstractVector{<:Point}))
  @eval begin

  function Arrays.return_cache(a::TempVec{D},x::$T) where D
    cf,ck = return_cache(a.a[1],x)
    Ck = Vector{typeof(cf)}(undef,D)
    r = return_cache(TensorProductMap(D),Ck)
    return cf,ck,Ck,r
  end

  function Arrays.evaluate!(cache,a::TempVec{D},x::$T) where D
    cf,ck,Ck,r = cache
    @inbounds for i = 1:D
      Ck[i] = evaluate!((cf,ck),a.a[i],x)
    end
    evaluate!(r,TensorProductMap(D),Ck)
  end

  end
end

for op in (:(Fields.∇),:(Fields.∇∇))
  @eval begin
    function Arrays.evaluate!(cache,k::Broadcasting{typeof($op)},a::TempVec)
      aop = ntuple(d->evaluate!(cache,k,a.a[d]),D)
      TempVec(aop)
    end
  end
end

############## test ##############
D = 2
T = VectorValue{2,Float64}
order = 2

reffe = ReferenceFE(QUAD,FEM.tplagrangian,T,order)
shapes = TempVec(get_shapefuns.(reffe.reffe))
dof_basis = get_dof_basis.(reffe.reffe)

prebasis1 = get_prebasis(reffe.reffe[1])
prebasis2 = get_prebasis(reffe.reffe[2])
change1 = inv(evaluate(dof_basis[1],prebasis1))
change2 = inv(evaluate(dof_basis[2],prebasis2))

tpreffe = ReferenceFE(QUAD,lagrangian,T,order)
tpshapes = get_shapefuns(tpreffe)
tpdof_basis = get_dof_basis(tpreffe)
tpprebasis = get_prebasis(tpreffe)

cf = return_cache(field,b.nodes)
vals = evaluate!(cf,field,b.nodes)

tpchange = inv(evaluate(tpdof_basis,tpprebasis))

Base.promote_rule(::Type{VectorValue{1,T}},::Val{D}) where D = VectorValue{D,T}

struct TensorProductMonomialBasis{D,T} <: AbstractVector{Monomial}
  bases::NTuple{D,MonomialBasis{1,T}}
  basis::MonomialBasis{D,T}
end

function TensorProductMonomialBasis(bases::NTuple{D,MonomialBasis{1,T}}) where {D,T}
  get_inner_terms(b::MonomialBasis) = b.terms
  _terms = get_inner_terms.(bases)
  nterms = map(length,_terms)
  terms = collect1d(CartesianIndices(nterms))
  orders = ntuple(i->get_order(bases[i]),D)
  basis = MonomialBasis{D}(T,orders,terms)
  TensorProductMonomialBasis(bases,basis)
end

Base.size(a::TensorProductMonomialBasis{D,T}) where {D,T} = size(a.basis)
Base.getindex(a::TensorProductMonomialBasis,i::Integer) = Monomial()
Base.IndexStyle(::TensorProductMonomialBasis) = IndexLinear()

ReferenceFEs.get_order(a::TensorProductMonomialBasis) = get_order(a.basis)
ReferenceFEs.get_orders(a::TensorProductMonomialBasis) = get_orders(a.basis)
ReferenceFEs.get_exponents(a::TensorProductMonomialBasis) = get_exponents(a.basis)

function Arrays.return_cache(f::TensorProductMonomialBasis{D,T},x::AbstractVector{<:Point}) where {D,T}
  return_cache(f.basis,x)
end

function Arrays.evaluate!(cache,f::TensorProductMonomialBasis{D,T},x::AbstractVector{<:Point}) where {D,T}
  evaluate!(cache,f.basis,x)
end

function Arrays.return_cache(
  fg::FieldGradientArray{Ng,TensorProductMonomialBasis{D,T}},
  x::AbstractVector{<:Point}) where {Ng,D,T}

  return_cache(FieldGradientArray{Ng}(fg.fa.basis),x)
end

function Arrays.evaluate!(
  cache,
  fg::FieldGradientArray{Ng,TensorProductMonomialBasis{D,T}},
  x::AbstractVector{<:Point}) where {Ng,D,T}

  evaluate!(cache,FieldGradientArray{Ng}(fg.fa.basis),x)
end

struct TensorProductDofBases{D,T,A,B} <: AbstractVector{T}
  dof_bases::NTuple{D,A}
  nodes::B

  function TensorProductDofBases(
    dof_bases::NTuple{D,A},
    nodes::TensorProductNodes{D}) where {D,T1<:Dof,A<:AbstractVector{T}}

    T = promote_rule(T1,Val(D))
    B = typeof(nodes)
    new{D,T,A,B}(dof_bases,nodes)
  end
end

Base.length(a::TensorProductDofBases) = length(a.nodes)
Base.size(a::TensorProductDofBases) = (length(a),)
Base.axes(a::TensorProductDofBases) = (Base.OneTo(length(a)),)
Base.IndexStyle(::TensorProductDofBases) = IndexLinear()
Base.getindex(a::TensorProductDofBases,i::Integer) = PointValue(a.nodes[i])

ReferenceFEs.get_nodes(a::TensorProductDofBases) = a.nodes

TensorValues.num_components(::TensorProductDofBases{D,T,LagrangianDofBasis{T1,V}}) where {D,T,T1,V} = num_components(V)

function ReferenceFEs.get_dof_to_node(a::TensorProductDofBases{D,T,<:LagrangianDofBasis}) where {D,T}
  nnodes = length(a.nodes)
  ncomps = num_components(a)
  ids = ntuple(i->collect(Base.OneTo(nnodes)),ncomps)
  collect1d(ids)
end

function ReferenceFEs.get_dof_to_comp(a::TensorProductDofBases{D,T,<:LagrangianDofBasis}) where {D,T}
  nnodes = length(a.nodes)
  ncomps = num_components(a)
  ids = ntuple(i->fill(i,nnodes),ncomps)
  collect1d(ids)
end

function ReferenceFEs.get_node_and_comp_to_dof(a::TensorProductDofBases{D,T,<:LagrangianDofBasis}) where {D,T}
  nnodes = length(a.nodes)
  ncomps = num_components(a)
  dof_to_node = get_dof_to_node(a)
  dof_to_comp = get_dof_to_comp(a)
  ids = (dof_to_comp .- 1)*ncomps .+ dof_to_node
  rids = reshape(ids,nnodes,ncomps)
  [VectorValue[rids[i,:]] for i = axes(rids,1)]
end

function Arrays.return_cache(a::TensorProductDofBases{D,T,<:LagrangianDofBasis},field) where {D,T}
  dof_to_node = get_dof_to_node(a)
  node_and_comp_to_dof = get_node_and_comp_to_dof(a)
  cf = return_cache(field,a.nodes)
  vals = evaluate!(cf,field,a.nodes)
  ndofs = length(dof_to_node)
  r = ReferenceFEs._lagr_dof_cache(vals,ndofs)
  c = CachedArray(r)
  (c,cf,dof_to_node,node_and_comp_to_dof)
end

function Arrays.evaluate!(cache,a::TensorProductDofBases{D,T,<:LagrangianDofBasis},field) where {D,T}
  c,cf,dof_to_node,node_and_comp_to_dof = cache
  vals = evaluate!(cf,field,a.nodes)
  ndofs = length(dof_to_node)
  S = eltype(vals)
  ncomps = num_components(S)
  @check ncomps == num_components(eltype(node_and_comp_to_dof)) """\n
  Unable to evaluate LagrangianDofBasis. The number of components of the
  given Field does not match with the LagrangianDofBasis.

  If you are trying to interpolate a function on a FESpace make sure that
  both objects have the same value type.

  For instance, trying to interpolate a vector-valued function on a scalar-valued FE space
  would raise this error.
  """
  ReferenceFEs._evaluate_lagr_dof!(c,vals,node_and_comp_to_dof,ndofs,ncomps)
end

function Arrays.return_cache(
  a::TensorProductDofBases{D,T,<:LagrangianDofBasis},
  field::TensorProductField) where {D,T}

  b = first(a.dof_bases)
  f = first(field.field)
  (c,cf,dof_to_node,node_and_comp_to_dof) = return_cache(b,f)
  vc = Vector{typeof(get_array(c))}(undef,D)
  C = _tp_lagr_dof_cache(c,length(get_dof_to_node(a)))
  (C,vc,c,cf,dof_to_node,node_and_comp_to_dof)
end

function Arrays.evaluate!(
  cache,
  a::TensorProductDofBases{D,T,<:LagrangianDofBasis},
  field::TensorProductField) where {D,T}

  C,vc,_cache... = cache
  @inbounds for d = 1:D
    vc[d] = evaluate!(_cache,a.dof_bases[d],field.field[d])
  end
  copyto!(C,kronecker(vc...))
  return C
end

function Arrays.return_cache(a::TensorProductDofBases{D,T,<:MomentBasedDofBasis},field) where {D,T}
  @notimplemented
end

function Arrays.evaluate!(cache,a::TensorProductDofBases{D,T,<:MomentBasedDofBasis},field) where {D,T}
  @notimplemented
end

struct TensorProductLinCombVec{D,V,F} <: AbstractVector{LinearCombinationFieldVector{V,F}}
  lincombs::NTuple{D,LinearCombinationFieldVector{V,F}}
  function TensorProductLinCombVec(lincombs::NTuple{D,LinearCombinationFieldVector{V,F}}
    ) where {D,V,F}
    item = testitem(lincombs)
    @check all(ntuple(i->size(a.lincombs[i],2) == size(item,2),D))
    new{D,V,F}(lincombs)
  end
end

Base.length(a::TensorProductLinComb) = size(first(a.lincombs),2)
Base.size(a::TensorProductLinComb) = (length(a),)
Base.IndexStyle(::Type{<:TensorProductLinComb}) = IndexLinear()
Base.getindex(a::TensorProductLinComb,i::Integer) = getindex(a.tplincomb,i)

function ReferenceFEs.compute_shapefuns(
  dofs::TensorProductDofBases{D},
  prebasis::TensorProductMonomialBasis{D}) where D

  changes = ntuple(i->inv(evaluate(dofs.dof_bases[i],prebasis.bases[i])),D)
  lincombs = ntuple(i->linear_combination(changes[i],prebasis.bases[i]),D)
  TensorProductLinComb(lincombs)
end
