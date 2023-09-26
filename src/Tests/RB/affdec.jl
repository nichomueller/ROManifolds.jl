# collect_compress_rhs(info,feop,fesolver,rbspace,snaps,params)
nsnaps = info.nsnaps_system
times = get_times(fesolver)
θ = fesolver.θ

nsnaps = info.nsnaps_system
_snaps,_params = get_at_params(1:nsnaps,snaps),params[1:nsnaps]
_snapsθ = center_solution(fesolver,_snaps,_params)
ress,meas = collect_residuals(feop,fesolver,_snapsθ,params;nsnaps)
ad_res = compress_component(info,feop,ress,meas,times,rbspace)

njacs = length(feop.jacs)
ad_jacs = Vector{RBAlgebraicContribution}(undef,njacs)
for i = 1:njacs
  combine_projections = (x,y) -> i == 1 ? θ*x+(1-θ)*y : x-y
  jacs,meas = collect_jacobians(feop,fesolver,args...;i,nsnaps)
  ad_jacs[i] = compress_component(info,feop,jacs,meas,times,rbspace,rbspace;combine_projections)
end


# ress,meas = collect_residuals(fesolver,feop,snaps,params;nsnaps=10)
op = feop
ode_op = get_algebraic_operator(op)
uu0 = get_free_dof_values(uh0params(params))
snapsθ = snaps*θ + [uu0,snaps[2:end]...]*(1-θ)
ptsnapsθ = to_ptarray(PTArray{Vector{Float}},snapsθ)
ode_cache = allocate_cache(ode_op,params,times)
ode_cache = update_cache!(ode_cache,ode_op,params,times)
nlop = PThetaMethodNonlinearOperator(ode_op,params,times,dt*θ,ptsnapsθ,ode_cache,ptsnapsθ)
printstyled("Computing fe residuals for every time and parameter\n";color=:blue)
ress = residual(nlop,ptsnapsθ)


# JACOBIAN
# nnz_jacs = map(eachindex(op.jacs)) do i
#   printstyled("Computing fe jacobian #$i for every time and parameter\n";color=:blue)
#   jacs_i = jacobian(nlop,ptsnapsθ,i)
#   nnz_jacs_i = map(NnzVector,jacs_i)
#   NnzMatrix(nnz_jacs_i)
# end
