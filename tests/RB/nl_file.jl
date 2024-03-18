function nl_jacobian_and_residual(solver::RB.ThetaMethodRBSolver,op::RBOperator{C},s) where C
  fesolver = RB.get_fe_solver(solver)
  dt = fesolver.dt
  θ = fesolver.θ
  θ == 0.0 ? dtθ = dt : dtθ = dt*θ
  r = get_realization(s)
  FEM.shift_time!(r,dt*(θ-1))
  ode_cache = allocate_cache(op,r)
  u0 = copy(get_values(s))
  vθ = similar(u0)
  vθ .= 0.0
  ode_cache = update_cache!(ode_cache,op,r)
  nlop = RBThetaMethodParamOperator(op,r,dtθ,u0,ode_cache,vθ)
  A = allocate_jacobian(nlop,u0)
  b = allocate_residual(nlop,u0)
  sA = jacobian!(A,nlop,u0)
  sb = residual!(b,nlop,u0)
  sA,sb
end

function nl_jacobian_and_residual(solver::ThetaMethod,op::ODEParamOperator{C},s) where C
  dt = solver.dt
  θ = solver.θ
  θ == 0.0 ? dtθ = dt : dtθ = dt*θ
  r = get_realization(s)
  FEM.shift_time!(r,dt*(θ-1))
  ode_cache = allocate_cache(op,r)
  u0 = copy(get_values(s))
  vθ = similar(u0)
  vθ .= 0.0
  ode_cache = update_cache!(ode_cache,op,r)
  nlop = ThetaMethodParamOperator(op,r,dtθ,u0,ode_cache,vθ)
  A = allocate_jacobian(nlop,u0)
  b = allocate_residual(nlop,u0)
  jacobian!(A,nlop,u0)
  residual!(b,nlop,u0)
  sA = map(A->Snapshots(A,r),A)
  sb = Snapshots(b,r)
  sA,sb
end

function nl_linear_combination_error(solver,feop,rbop,s)
  feA,feb = nl_jacobian_and_residual(RB.get_fe_solver(solver),get_algebraic_operator(feop),s)
  feA_comp = compress(solver,feA,get_trial(rbop),get_test(rbop))
  feb_comp = compress(solver,feb,get_test(rbop))
  rbA,rbb = nl_jacobian_and_residual(solver,rbop,s)
  errA = RB._rel_norm(feA_comp,rbA)
  errb = RB._rel_norm(feb_comp,rbb)
  return errA,errb
end

son = select_snapshots(fesnaps,51)
ron = get_realization(son)
θ == 0.0 ? dtθ = dt : dtθ = dt*θ

r = copy(ron)
FEM.shift_time!(r,dt*(θ-1))

rb_trial = get_trial(rbop)(r)
fe_trial = get_fe_trial(rbop)(r)
red_x = zero_free_values(rb_trial)
y = zero_free_values(fe_trial)
z = similar(y)
z .= 0.0

ode_cache = allocate_cache(rbop,r)
cache_lin = ODETools._allocate_matrix_and_vector(rbop.op_linear,r,y,ode_cache)
cache_nlin = ODETools._allocate_matrix_and_vector(rbop.op_nonlinear,r,y,ode_cache)
cache = cache_lin,cache_nlin

ode_cache = update_cache!(ode_cache,rbop,r)
nlop = RBThetaMethodParamOperator(rbop,r,dtθ,y,ode_cache,z)
# solve!(red_x,fesolver.nls,nlop,cache)

fex = copy(nlop.u0)
(cache_jac_lin,cache_res_lin),(cache_jac_nlin,cache_res_nlin) = cache

# linear res/jac, now they are treated as cache
lop = nlop.odeop.op_linear
A_lin,b_lin = ODETools._matrix_and_vector!(cache_jac_lin,cache_res_lin,lop,r,dtθ,y,ode_cache,z)

# initial nonlinear res/jac
nnlop = nlop.odeop.op_nonlinear
b_nlin = residual!(cache_res_nlin,nnlop,r,(fex,z),ode_cache)
A_nlin = ODETools.jacobians!(cache_jac_nlin,nnlop,r,(fex,z),(1,1/dtθ),ode_cache)
A = A_nlin + A_lin
b = b_nlin - b_lin
dx = similar(b)
ss = symbolic_setup(LUSolver(),A)
ns = numerical_setup(ss,A)

trial = get_trial(nlop.odeop)(nlop.r)
isconv, conv0 = Algebra._check_convergence(nls,b)

######### loop ##########
rmul!(b,-1)
solve!(dx,ns,b)
red_x .+= dx

fex = recast(red_x,trial)
# eA,eB = nl_linear_combination_error(rbsolver,feop.op_nonlinear,rbop.op_nonlinear,Snapshots(fex,r))
feA,feb = nl_jacobian_and_residual(fesolver,get_algebraic_operator(feop.op_nonlinear),Snapshots(fex,r))
feA_comp = compress(rbsolver,feA,get_trial(rbop),get_test(rbop))
feb_comp = compress(rbsolver,feb,get_test(rbop))
rbA,rbb = nl_jacobian_and_residual(rbsolver,rbop.op_nonlinear,Snapshots(fex,r))

b_nlin = residual!(cache_res_nlin,nnlop.odeop,r,(fex,),ode_cache)
b = -b_lin + b_nlin
isconv = Algebra._check_convergence(nls,b,conv0)
println(maximum(abs,b))

A_nlin = jacobian!(cache_jac_nlin,nnlop.odeop,r,(fex,),ode_cache)
A = A_lin + A_nlin
numerical_setup!(ns,A)
######### loop ##########
