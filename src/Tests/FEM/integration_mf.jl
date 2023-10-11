μ = realization(feop)
t = dt
row,col = 1,2
feop_row_col = feop[row,col]
u = zero(feop_row_col.test)
du = get_trial_fe_basis(feop_row_col.trials[1](nothing,nothing))
dv = get_fe_basis(feop_row_col.test)
integrate(feop_row_col.jacs[1](μ,t,u,du,dv),DomainContribution())
integrate(feop_row_col.jacs[2](μ,t,u,du,dv),DomainContribution())

dṽ,dũ,ũ = [],[],[]
for n = 1:2
  push!(dṽ,nothing)
  push!(dũ,nothing)
  push!(ũ,nothing)
end
dṽ[row] = dv
dũ[col] = du
ũ[col] = u
jjac(μ,t,(u,p),(du,dp),(v,q)) = ∫(aμt(μ,t)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ - ∫(q*(∇⋅(du)))dΩ
jjac(realization(feop),dt,ũ,dũ,dṽ)

μ = realization(feop,2)
times = [dt,2*dt]
row,col = 1,1
feop_row_col = feop[row,col]

V = get_test(feop_row_col)
dv = get_fe_basis(V)
U = get_trial(feop_row_col)(nothing,nothing)
du = get_trial_fe_basis(U)
# V = get_test(feop)
# dv = get_fe_basis(V)
# U = get_trial(feop)(nothing,nothing)
# du = get_trial_fe_basis(U)

ode_op = get_algebraic_operator(feop_row_col)
ode_cache = allocate_cache(ode_op,μ,times)
update_cache!(ode_cache,ode_op,μ,times)
nfree = num_free_dofs(V)
u = PTArray([zeros(V) for _ = 1:length(μ)*length(times)])
vθ = similar(u)
vθ .= 1.0
Us,_,fecache = ode_cache
uh = EvaluationFunction(Us[1],vθ)
dxh = ()
for i in 1:get_order(feop_row_col)
  dxh = (dxh...,uh)
end
xh = TransientCellField(uh,dxh)

# res(μ,t,(u,p),(v,q)) = (∫ₚ(v⋅∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⊙∇(u),dΩ) - ∫ₚ(p*(∇⋅(v)),dΩ)
# - ∫ₚ(q*(∇⋅(u)),dΩ) - ∫ₚ(v⋅fμt(μ,t),dΩ))
empty_dv,empty_du,empty_u = Any[],Any[],Any[]
for n in eachindex(get_test(feop).spaces)
  push!(empty_dv,nothing)
  push!(empty_du,nothing)
  push!(empty_u,nothing)
end

function u_col(u)
  empty_u[col] = u
  return empty_u
end
function du_col(du)
  empty_du[col] = du
  return empty_du
end
function dv_row(dv)
  empty_dv[row] = dv
  return empty_dv
end

dcr = integrate(feop_row_col.res(μ,times,xh,dv))
dcr_manual = integrate(∫ₚ(dv⋅∂ₚt(xh),dΩ) + ∫ₚ(aμt(μ,times)*∇(dv)⊙∇(xh),dΩ))

ũ = (xh,nothing)#u_col(xh)
dṽ = (dv,nothing)#dv_row(dv)
feop.res(μ,times,ũ,dṽ)

int = feop_row_col.res(μ,times,xh,dv)
# integrate(int.object,int.meas)
# c = integrate(int.object,int.meas.quad)
b = change_domain(int.object,int.meas.quad.trian,int.meas.quad.data_domain_style)
x = get_cell_points(int.meas.quad)
bx = b(x)


# collect_cell_vector(V,dcr)
