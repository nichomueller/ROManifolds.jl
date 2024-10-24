# tests

using ReducedOrderModels.ParamFESpaces
using ReducedOrderModels.ParamDataStructures

# r = realization(ptspace;nparams=1)
# trial(r)
Upt = allocate_space(trial,r)
# evaluate!(Upt,trial,r)
μ,t = get_params(r),get_times(r)
dir(f) = f(μ,t)
dir(f::Vector) = dir.(f)
# TrialParamFESpace!(Upt,dir(trial.dirichlet))
objects = dir(trial.dirichlet)
dir_values = get_dirichlet_dof_values(Upt)
dir_values_scratch = zero_dirichlet_values(Upt)
# dir_values = compute_dirichlet_values_for_tags!(dir_values,dir_values_scratch,Upt,objects)
dirichlet_dof_to_tag = get_dirichlet_dof_tag(Upt)
_tag_to_object = FESpaces._convert_to_collectable(objects,num_dirichlet_tags(Upt))
(tag,object) = 1,_tag_to_object[1]
cell_vals = FESpaces._cell_vals(Upt,object)
fill!(dir_values_scratch,zero(eltype(dir_values_scratch)))
# gather_dirichlet_values!(dir_values_scratch,Upt,cell_vals)
cell_dofs = get_cell_dof_ids(Upt)
cache_vals = array_cache(cell_vals)
cache_dofs = array_cache(cell_dofs)
free_vals = zero_free_values(Upt)
cells = ParamFESpaces.get_dirichlet_cells(Upt)

dirichlet_vals = dir_values_scratch
free_data = get_all_data(free_vals)
diri_data = get_all_data(dir_values_scratch)
free_data = get_all_data(free_vals)
diri_data = get_all_data(dirichlet_vals)
cell = cells[1]
vals = getindex!(cache_vals,cell_vals,cell)
dofs = getindex!(cache_dofs,cell_dofs,cell)
(i,dof) = 1,dofs[1]
k = 10
val = vals[k][i]
if dof > 0
  free_data[dof,k] = val
elseif dof < 0
  diri_data[-dof,k] = val
end
# FESpaces._fill_dirichlet_values_for_tag!(dir_values,dir_values_scratch,tag,dirichlet_dof_to_tag)

using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData
using Gridap.MultiField
using Gridap.ODEs
using Gridap.TensorValues
using Gridap.Helpers

using ReducedOrderModels.ParamFESpaces

# odeop = get_algebraic_operator(feop.op)

# us = (zero_free_values(trial(r)),zero_free_values(trial(r)))

# order = get_order(odeop)
# pttrial = get_trial(odeop.op)
# trial = allocate_space(pttrial,r)
# pttrials = (pttrial,)
# trials = (trial,)
# for k in 1:order
#   pttrials = (pttrials...,∂t(pttrials[k]))
#   trials = (trials...,allocate_space(pttrials[k+1],r))
# end

# feop_cache = allocate_feopcache(odeop.op,r,us)

# uh = ODEs._make_uh_from_us(odeop,us,trials)
# test = get_test(odeop.op)
# v = get_fe_basis(test)
# du = get_trial_fe_basis(get_trial(odeop.op)(nothing))
# assem = get_param_assembler(odeop.op,r)

# const_forms = ()
# num_forms = get_num_forms(odeop.op)
# jacs = get_jacs(odeop.op)

# μ,t = get_params(r),get_times(r)

# dc = DomainContribution()
# for k in 1:order+1
#   jac = jacs[k]
#   dc = dc + jac(μ,t,uh,du,v)
# end
# matdata = collect_cell_matrix(trial,test,dc)
# A_full = allocate_matrix(assem,matdata)
# assemble_matrix_add!(A_full,assem,matdata)

# m1 = nz_counter(get_matrix_builder(assem),(get_rows(assem),get_cols(assem)))
# symbolic_loop_matrix!(m1,assem,matdata)

# m2 = nz_allocation(m1)
# symbolic_loop_matrix!(m2,assem,matdata)




# ################### gridap

# assem′ = get_assembler(odeop.op)
# dc′ = ∫(a(μ.params[1],t[1])*∇(v)⋅∇(du))dΩ
# matdata′ = collect_cell_matrix(trial,test,dc′)

# m1′ = nz_counter(get_matrix_builder(assem′),(get_rows(assem′),get_cols(assem′)))
# symbolic_loop_matrix!(m1′,assem′,matdata′)
# m2′ = nz_allocation(m1′)
# symbolic_loop_matrix!(m2′,assem′,matdata′)

# ##############################

using ReducedOrderModels.ParamDataStructures

solver = fesolver
odeop = get_algebraic_operator(feop.op)
r0 = ParamDataStructures.get_at_time(r,0.05)
us0 = (fill!(zero_free_values(trial(r0)),1.0),)
odeparamcache = allocate_odeparamcache(solver,odeop,r0,us0)
state0,cache = ode_start(solver,odeop,r0,us0,odeparamcache)

(odeslvrcache,paramcache) = odeparamcache
(A,b,sysslvrcache) = odeslvrcache

w0 = state0[1]
odeslvrcache,paramcache = odeparamcache
A,b,sysslvrcache = odeslvrcache
sysslvr = solver.sysslvr
dt,θ = solver.dt,solver.θ
x = us0[1]
fill!(x,one(eltype(x)))
dtθ = θ*dt
# shift!(r0,dtθ)
usx = (w0,x)
ws = (dtθ,1)
update_paramcache!(paramcache,odeop,r0)
stageop = LinearParamStageOperator(odeop,paramcache,r0,usx,ws,A,b,sysslvrcache)
# sysslvrcache = solve!(x,sysslvr,stageop,sysslvrcache)

uh = ODEs._make_uh_from_us(odeop,usx,paramcache.trial)
v = get_fe_basis(test)
assem = get_param_assembler(odeop.op,r0)
b = allocate_residual(odeop,r0,usx,paramcache)
μ,t = get_params(r0),get_times(r0)
# Residual
resf = get_res(odeop.op)
dc = resf(μ,t,uh,v)
vecdata = collect_cell_vector(test,dc)
assemble_vector_add!(b,assem,vecdata)

dc1 = ∫(v*∂t(uh))dΩ
dc2 = ∫(aμt(μ,t)*∇(v)⋅∇(uh))dΩ
dc3 = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn


######################################
# FESpaces._cell_vals(Upt,object)
s = get_fe_dof_basis(Upt)
trian = get_triangulation(s)
cf = CellField(object,trian,DomainStyle(s))
# s(cf)
b = change_domain(cf,s.domain_style)
# lazy_map(evaluate,get_data(s),get_data(b))
# evaluate(get_data(s)[1],get_data(b)[1])
cache = return_cache(get_data(s)[1],get_data(b)[1])
b,field = get_data(s)[1],get_data(b)[1]
cf = return_cache(field,b.nodes)
vals = evaluate!(cf,field,b.nodes)
ndofs = length(b.dof_to_node)
r = ReferenceFEs._lagr_dof_cache(vals,ndofs)
c = CachedArray(r)
