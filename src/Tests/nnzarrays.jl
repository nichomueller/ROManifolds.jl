nsnaps = 10
op,solver = feop,fesolver
μ = realization(op,nsnaps)
ode_op = get_algebraic_operator(op)
_u0 = get_free_dof_values(uh0μ(μ))
uμt = PODESolution(solver,ode_op,μ,_u0,t0,tf)
num_iter = Int(tf/solver.dt)
solutions = allocate_solution(ode_op,num_iter)
for (u,t,n) in uμt
  printstyled("Computing fe solution at time $t for every parameter\n";color=:blue)
  solutions[n] = copy(get_solution(ode_op,u))
end
nparams=length(testitem(solutions))
@check all([length(vali) == nparams for vali in solutions])
vals = hcat(get_array(hcat(solutions...))...)
nonzero_idx,nonzero_val = compress_array(vals)
nrows = size(vals,1)
snaps = NnzMatrix(nonzero_val,nonzero_idx,nrows,nparams)
ϵ = info.ϵ
energy_norm = info.energy_norm
norm_matrix = get_norm_matrix(energy_norm,feop)
steady = num_time_dofs(snaps) == 1 ? SteadyPOD() : DefaultPOD()
transposed = size(snaps,1) < size(snaps,2) ? TranposedPOD() : DefaultPOD()
snaps = compress(snaps,norm_matrix,steady,transposed;ϵ)

sols = solutions
dt,θ = solver.dt,solver.θ
dtθ = θ == 0.0 ? dt : dt*θ
times = collect(dt:dt:(tf-dt) .+ dtθ)
ode_op = get_algebraic_operator(op)
uu0 = get_free_dof_values(uh0μ(μ))
solsθ = copy(sols)
solsθ .= sols*θ + [uu0,sols[2:end]...]*(1-θ)
ptsolsθ = convert(PTArray,solsθ)#PTArray(vcat(map(get_array,solsθ)...))
ode_cache = allocate_cache(ode_op,μ,times)
ode_cache = update_cache!(ode_cache,ode_op,μ,times)
nlop = PThetaMethodNonlinearOperator(ode_op,μ,times,dtθ,ptsolsθ,ode_cache,ptsolsθ)
printstyled("Computing fe residuals for every time and parameter\n";color=:blue)
ress = residual(nlop,ptsolsθ)


nnz_jacs = map(eachindex(op.jacs)) do i
  printstyled("Computing fe jacobian #$i for every time and parameter\n";color=:blue)
  jacs_i = jacobian(nlop,ptsolsθ,i)
  nnz_jacs_i = map(NnzVector,jacs_i)
  NnzMatrix(nnz_jacs_i)
end
