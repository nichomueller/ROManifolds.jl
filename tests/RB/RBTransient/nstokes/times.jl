using Gridap
using Gridap.Algebra
using Gridap.CellData
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.ODEs
using Test
using DrWatson
using Serialization
using BenchmarkTools

using ReducedOrderModels
using ReducedOrderModels.ParamDataStructures
using ReducedOrderModels.ParamSteady
using ReducedOrderModels.ParamFESpaces

# time marching
θ = 0.5
dt = 0.0025
t0 = 0.0
tf = 60*dt

# parametric space
pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

order = 2
degree = 2*order+1

const Re′ = 100.0
a(x,μ,t) = μ[1]/Re′
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

const W = 0.5
inflow(μ,t) = abs(1-cos(2π*t/tf)+μ[3]*sin(μ[2]*2π*t/tf)/100)
g_in(x,μ,t) = VectorValue(-x[2]*(W-x[2])*inflow(μ,t),0.0,0.0)
g_in(μ,t) = x->g_in(x,μ,t)
gμt_in(μ,t) = TransientParamFunction(g_in,μ,t)
g_0(x,μ,t) = VectorValue(0.0,0.0,0.0)
g_0(μ,t) = x->g_0(x,μ,t)
gμt_0(μ,t) = TransientParamFunction(g_0,μ,t)

u0(x,μ) = VectorValue(0.0,0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0.0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

conv(u,∇u) = (∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

# loop on mesh
for h in ("h007","h005","h0035")
  model_dir = datadir(joinpath("models","model_circle_$h.json"))
  model = DiscreteModelFromFile(model_dir)
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"dirichlet_noslip",["cylinders","walls"])
  add_tag_from_tags!(labels,"dirichlet_nopenetration",["top_bottom"])
  add_tag_from_tags!(labels,"dirichlet",["inlet"])

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
  dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

  djac(μ,t,(uₜ,pₜ),(dut,dpt),(v,q)) = ∫(v⋅dut)dΩ
  jac(μ,t,(u,p),(du,dp),(v,q)) = dc(u,du,v) + ∫(aμt(μ,t)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ - ∫(q*(∇⋅(du)))dΩ
  res(μ,t,(u,p),(v,q)) = c(u,v) + ∫(v⋅∂t(u))dΩ + ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ

  reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  test_u = TestFESpace(model,reffe_u;conformity=:H1,
    dirichlet_tags=["inlet","dirichlet_nopenetration","dirichlet_noslip"],
    dirichlet_masks=[[true,true,true],[false,false,true],[true,true,true]])
  trial_u = TransientTrialParamFESpace(test_u,[gμt_in,gμt_in,gμt_0])
  reffe_p = ReferenceFE(lagrangian,Float64,order-1)
  test_p = TestFESpace(model,reffe_p;conformity=:C0)
  trial_p = TrialFESpace(test_p)
  test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
  trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
  feop = TransientParamFEOperator(res,(jac,djac),ptspace,trial,test)

  xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

  odeop = get_algebraic_operator(feop)
  ws = (1,1)
  us(x) = (x,x)

  # loop on params
  for nparams in 1:10
    r = realization(ptspace;nparams)

    U = trial(r)
    x = get_free_dof_values(zero(U))

    paramcache = allocate_paramcache(odeop,r,(x,x))
    stageop = NonlinearParamStageOperator(odeop,paramcache,r,us,ws)

    println("Residual time with h = $h, nparams = $nparams:")
    @btime residual!(allocate_residual($stageop,$x),$stageop,$x)

    println("Jacobian time with h = $h, nparams = $nparams:")
    @btime jacobian!(allocate_jacobian($stageop,$x),$stageop,$x)

    # println("Solve time with h = $h, nparams = $nparams:")
    # @btime solve!(x,LUSolver(),A,b)
  end
end

# Gridap
μ = rand(3)
g′_in(x,t) = VectorValue(-x[2]*(W-x[2])*inflow(μ,t),0.0,0.0)
g′_in(t) = x->g′_in(x,t)
g′_0(x,t) = VectorValue(0.0,0.0,0.0)
g′_0(t) = x->g′_0(x,t)

# loop on mesh
for h in ("h007","h005","h0035")
  model_dir = datadir(joinpath("models","model_circle_$h.json"))
  model = DiscreteModelFromFile(model_dir)
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"dirichlet_noslip",["cylinders","walls"])
  add_tag_from_tags!(labels,"dirichlet_nopenetration",["top_bottom"])
  add_tag_from_tags!(labels,"dirichlet",["inlet"])

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
  dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

  djac(t,(uₜ,pₜ),(dut,dpt),(v,q)) = ∫(v⋅dut)dΩ
  jac(t,(u,p),(du,dp),(v,q)) = dc(u,du,v) + ∫(a(μ,t)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ - ∫(q*(∇⋅(du)))dΩ
  res(t,(u,p),(v,q)) = c(u,v) + ∫(v⋅u)dΩ + ∫(a(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ

  reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  test_u = TestFESpace(model,reffe_u;conformity=:H1,
    dirichlet_tags=["inlet","dirichlet_nopenetration","dirichlet_noslip"],
    dirichlet_masks=[[true,true,true],[false,false,true],[true,true,true]])
  trial_u = TransientTrialFESpace(test_u,[g′_in,g′_in,g′_0])
  reffe_p = ReferenceFE(lagrangian,Float64,order-1)
  test_p = TestFESpace(model,reffe_p;conformity=:C0)
  trial_p = TrialFESpace(test_p)
  test = TransientMultiFieldFESpace([test_u,test_p];style=BlockMultiFieldStyle())
  trial = TransientMultiFieldFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
  feop = TransientFEOperator(res,(jac,djac),trial,test)

  xh0 = interpolate_everywhere([u0(μ),p0(μ)],trial(t0))

  odeop = get_algebraic_operator(feop)
  ws = (1,1)
  us(x) = (x,x)

  U = trial(t0)
  x = get_free_dof_values(zero(U))

  odeopcache = allocate_odeopcache(odeop,t0,(x,x))
  stageop = NonlinearStageOperator(odeop,odeopcache,t0,us,ws)

  println("Residual time with h = $h:")
  @btime residual!(allocate_residual($stageop,$x),$stageop,$x)

  println("Jacobian time with h = $h:")
  @btime jacobian!(allocate_jacobian($stageop,$x),$stageop,$x)

  # println("Solve time with h = $h:")
  # @btime solve!(x,LUSolver(),A,b)
end


h ="h0035"
model_dir = datadir(joinpath("models","model_circle_$h.json"))
model = DiscreteModelFromFile(model_dir)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet_noslip",["cylinders","walls"])
add_tag_from_tags!(labels,"dirichlet_nopenetration",["top_bottom"])
add_tag_from_tags!(labels,"dirichlet",["inlet"])

Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

djac(μ,t,(uₜ,pₜ),(dut,dpt),(v,q)) = ∫(v⋅dut)dΩ
jac(μ,t,(u,p),(du,dp),(v,q)) = dc(u,du,v) + ∫(aμt(μ,t)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ - ∫(q*(∇⋅(du)))dΩ
res(μ,t,(u,p),(v,q)) = c(u,v) + ∫(v⋅∂t(u))dΩ + ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,
  dirichlet_tags=["inlet","dirichlet_nopenetration","dirichlet_noslip"],
  dirichlet_masks=[[true,true,true],[false,false,true],[true,true,true]])
trial_u = TransientTrialParamFESpace(test_u,[gμt_in,gμt_in,gμt_0])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:C0)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = TransientParamFEOperator(res,(jac,djac),ptspace,trial,test)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

odeop = get_algebraic_operator(feop)
ws = (1,1)
us(x) = (x,x)

nparams = 1
r = realization(ptspace;nparams)

U = trial(r)
x = get_free_dof_values(zero(U))

paramcache = allocate_paramcache(odeop,r,(x,x))

# A = allocate_jacobian(stageop,x)
usx = us(x)
# allocate_jacobian(odeop,rx,usx,paramcache)
uh = ODEs._make_uh_from_us(odeop,usx,paramcache.trial)
trial = evaluate(get_trial(odeop.op),nothing)
du = get_trial_fe_basis(trial)
test = get_test(odeop.op)
v = get_fe_basis(test)
assem = get_param_assembler(odeop.op,r)

μ,t = get_params(r),get_times(r)

jacs = get_jacs(odeop.op)
Dc = DomainContribution()
for k in 1:2
  Jac = jacs[k]
  Dc = Dc + Jac(μ,t,uh,du,v)
end
matdata = collect_cell_matrix(trial,test,Dc)
A = allocate_matrix(assem,matdata)

# jacobian!(A,stageop,x)
# assemble_matrix_add!(A,assem,matdata)
using BlockArrays
using Gridap.Algebra
using Gridap.Arrays
using Gridap.Fields

m1 = ArrayBlock(blocks(A),fill(true,blocksize(A)))
m2 = MultiField.expand_blocks(assem,m1)
# FESpaces.assemble_matrix_add!(m2,assem,matdata)
# numeric_loop_matrix!(m2,assem,matdata)
(cellmat,cellidsrows,cellidscols) = first(zip(matdata...))
rows_cache = array_cache(cellidsrows)
cols_cache = array_cache(cellidscols)
vals_cache = array_cache(cellmat)
mat1 = getindex!(vals_cache,cellmat,1)
rows1 = getindex!(rows_cache,cellidsrows,1)
cols1 = getindex!(cols_cache,cellidscols,1)
add! = FESpaces.AddEntriesMap(+)
add_cache = return_cache(add!,A,mat1,rows1,cols1)

for cell in 1:length(cellidscols)
  rows = getindex!(rows_cache,cellidsrows,cell)
  cols = getindex!(cols_cache,cellidscols,cell)
  vals = getindex!(vals_cache,cellmat,cell)
  evaluate!(add_cache,add!,m2,vals,rows,cols)
end

cell = 1
# vals = getindex!(vals_cache,cellmat,cell)
cache = vals_cache
_cache, index_and_item = cache
index = LinearIndices(cellmat)[cell...]
# if index_and_item.index != index
cg, cgi, cf = _cache
gi = getindex!(cg, cellmat.maps, cell...)
index_and_item.item = Arrays._getindex_and_call!(cgi,gi,cf,cellmat.args,cell...)
index_and_item.index = index


#############

μ = rand(3)
g′_in(x,t) = VectorValue(-x[2]*(W-x[2])*inflow(μ,t),0.0,0.0)
g′_in(t) = x->g′_in(x,t)
g′_0(x,t) = VectorValue(0.0,0.0,0.0)
g′_0(t) = x->g′_0(x,t)

c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

djac(t,(uₜ,pₜ),(dut,dpt),(v,q)) = ∫(v⋅dut)dΩ
jac(t,(u,p),(du,dp),(v,q)) = dc(u,du,v) + ∫(a(μ,t)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ - ∫(q*(∇⋅(du)))dΩ
res(t,(u,p),(v,q)) = c(u,v) + ∫(v⋅u)dΩ + ∫(a(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,
  dirichlet_tags=["inlet","dirichlet_nopenetration","dirichlet_noslip"],
  dirichlet_masks=[[true,true,true],[false,false,true],[true,true,true]])
trial_u = TransientTrialFESpace(test_u,[g′_in,g′_in,g′_0])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:C0)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = TransientFEOperator(res,(jac,djac),trial,test)

xh0 = interpolate_everywhere([u0(μ),p0(μ)],trial(t0))

odeop = get_algebraic_operator(feop)
ws = (1,1)
us(x) = (x,x)

U = trial(t0)
x = get_free_dof_values(zero(U))

paramcache = allocate_paramcache(odeop,t0,(x,x))
stageop = NonlinearStageOperator(odeop,paramcache,t0,us,ws)

usx = stageop.usx(x)
A = allocate_jacobian(odeop,t0,usx,paramcache)

uh = ODEs._make_uh_from_us(odeop,usx,paramcache.trial)
trial = evaluate(trial,nothing)
du = get_trial_fe_basis(trial)
v = get_fe_basis(test)
assem = get_assembler(feop)

Dc = DomainContribution()
for k in 1:2
  Jac = feop.jacs[k]
  Dc = Dc + Jac(t,uh,du,v)
end
matdata = collect_cell_matrix(trial,test,Dc)

# jacobian!(A,stageop,x)
# assemble_matrix_add!(A,assem,matdata)

m1 = ArrayBlock(blocks(A),fill(true,blocksize(A)))
m2 = MultiField.expand_blocks(assem,m1)
# FESpaces.assemble_matrix_add!(m2,assem,matdata)
# numeric_loop_matrix!(m2,assem,matdata)
(cellmat,cellidsrows,cellidscols) = first(zip(matdata...))
rows_cache = array_cache(cellidsrows)
cols_cache = array_cache(cellidscols)
vals_cache = array_cache(cellmat)
mat1 = getindex!(vals_cache,cellmat,1)
rows1 = getindex!(rows_cache,cellidsrows,1)
cols1 = getindex!(cols_cache,cellidscols,1)
add! = FESpaces.AddEntriesMap(+)
add_cache = return_cache(add!,A,mat1,rows1,cols1)

for cell in 1:length(cellidscols)
  rows = getindex!(rows_cache,cellidsrows,cell)
  cols = getindex!(cols_cache,cellidscols,cell)
  vals = getindex!(vals_cache,cellmat,cell)
  evaluate!(add_cache,add!,m2,vals,rows,cols)
end

# trial(r)
μ,t = get_params(r),get_times(r)
# evaluate(trial,μ,t)
U = trial
Upt = allocate_space(U,μ,t)
# evaluate!(Upt,U,μ,t)
# evaluate!(Upt[1],U[1],μ,t)
Upt = Upt[1]
U = U[1]
dir(f) = f(μ,t)
dir(f::Vector) = dir.(f)
# TrialParamFESpace!(Upt,dir(U.dirichlet))
objects = dir(U.dirichlet)
dir_values = get_dirichlet_dof_values(Upt)
dir_values_scratch = zero_dirichlet_values(Upt)
# dir_values = compute_dirichlet_values_for_tags!(dir_values,dir_values_scratch,Upt,objects)
dirichlet_dof_to_tag = get_dirichlet_dof_tag(Upt)
_tag_to_object = FESpaces._convert_to_collectable(objects,num_dirichlet_tags(Upt))
(tag, object) = 1,_tag_to_object[1]
cell_vals = FESpaces._cell_vals(Upt,object)
fill!(dir_values_scratch,zero(eltype(dir_values_scratch)))
# FESpaces.gather_dirichlet_values!(dir_values_scratch,Upt,cell_vals)
cell_dofs = get_cell_dof_ids(Upt)
cache_vals = array_cache(cell_vals)
cache_dofs = array_cache(cell_dofs)
free_vals = zero_free_values(Upt)
cells = ParamFESpaces.get_dirichlet_cells(Upt)

free_data = get_all_data(free_vals)
diri_data = get_all_data(dir_values_scratch)

cell = 1
@btime begin
  vals = getindex!($cache_vals,$cell_vals,1)
  dofs = getindex!($cache_dofs,$cell_dofs,1)
  for (i,dof) in enumerate(dofs)
    @inbounds for k in param_eachindex($free_vals)
      val = get_all_data(vals)[i,k]
      if dof > 0
        $free_data[dof,k] = val
      elseif dof < 0
        $diri_data[-dof,k] = val
      else
        @assert false
      end
    end
  end
end

@btime begin
  for cell in 1:6817
    getindex!($cache_vals,$cell_vals,cell)
  end
end
# FESpaces._fill_dirichlet_values_for_tag!(dir,dir_values_scratch,tag,dirichlet_dof_to_tag)

V′ = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["inlet"])
U′ = TransientTrialFESpace(V′,g′_in)

Ut′ = allocate_space(U′)
objects′ = U′.transient_dirichlet(t[1])
dir_values′ = get_dirichlet_dof_values(Ut′)
dir_values_scratch′ = zero_dirichlet_values(Ut′)
_tag_to_object′ = FESpaces._convert_to_collectable(objects′,1)
(tag′, object′) = 1,_tag_to_object′[1]
cell_vals′ = FESpaces._cell_vals(Ut′,objects′)
fill!(dir_values_scratch′,zero(eltype(dir_values_scratch′)))
@btime FESpaces.gather_dirichlet_values!($dir_values_scratch′,$Ut′,$cell_vals′)
cell_dofs′ = get_cell_dof_ids(Ut′)
cache_vals′ = array_cache(cell_vals′)
cache_dofs′ = array_cache(cell_dofs′)
free_vals′ = zero_free_values(Ut′)

@btime begin
  for cell in 1:6817
    getindex!($cache_vals′,$cell_vals′,cell)
  end
end

using Gridap.Arrays
using Gridap.ReferenceFEs

# getindex!(cache_vals,cell_vals,cell)
i = (cell,)
_cache, index_and_item = cache_vals
index = LinearIndices(cell_vals)[i...]
index_and_item.index != index
cg, cgi, cf = _cache
gi = getindex!(cg, cell_vals.maps, i...)
# index_and_item.item = Arrays._getindex_and_call!(cgi,gi,cf,cell_vals.args,i...)
f1 = getindex!(cf[1],cell_vals.args[1],i...)
# evaluate!(cgi,gi,f1)
cc, cf = cgi
vals = evaluate!(cf,f1,gi.nodes)
ndofs = length(gi.dof_to_node)
T = eltype(vals)
ncomps = num_components(T)
ReferenceFEs._evaluate_lagr_dof!(cc,vals,gi.node_and_comp_to_dof,ndofs,ncomps)

# getindex!(cache_vals′,cell_vals′,cell)
_cache′, index_and_item′ = cache_vals′
index′ = LinearIndices(cell_vals′)[i...]
index_and_item′.index != index′
cg′, cgi′, cf′ = _cache′
gi′ = getindex!(cg′, cell_vals′.maps, i...)
# index_and_item′.item = Arrays._getindex_and_call!(cgi′,gi′,cf′,cell_vals′.args,i...)
f1′ = getindex!(cf′[1],cell_vals′.args[1],i...)
evaluate!(cgi′,gi′,f1′)
