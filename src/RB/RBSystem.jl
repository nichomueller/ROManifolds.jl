function steady_poisson_rb_system(
  rbpos::NTuple{N,RBParamOnlineStructure},
  μ::Param) where N

  lhs = assemble(rbpos,:A,μ)
  rhs = assemble(rbpos,(:F,:H,:A_lift),μ)
  lhs,sum(rhs)
end

function unsteady_poisson_rb_system(
  rbpos::NTuple{N,RBParamOnlineStructure},
  μ::Param) where N

  lhs = assemble(rbpos,:A,μ)+assemble(rbpos,:M,μ)
  rhs = (assemble(rbpos,:F,μ)+assemble(rbpos,:H,μ)+
    assemble(rbpos,:A_lift,μ)+assemble(rbpos,:M_lift,μ))
  lhs,rhs
end

function steady_stokes_rb_system(
  rbpos::NTuple{N,RBParamOnlineStructure},
  μ::Param) where N

  lhs = assemble(rbpos,(:A,:B),μ)
  rhs = assemble(rbpos,(:F,:H,:A_lift,:B_lift),μ)

  np = size(lhs[2],1)
  rb_lhs = vcat(hcat(lhs[1],-lhs[2]'),hcat(lhs[2],zeros(np,np)))
  rb_rhs = vcat(sum(rhs[1:end-1]),rhs[end])
  rb_lhs,rb_rhs
end

function unsteady_stokes_rb_system(
  rbpos::NTuple{N,RBParamOnlineStructure},
  μ::Param) where N

  mblock11 = assemble(rbpos,:A,μ)+assemble(rbpos,:M,μ)
  mblock12 = -assemble(rbpos,:BT,μ)
  mblock21 = assemble(rbpos,:B,μ)
  mblock22 = zeros(size(mblock21,1),size(mblock12,2))

  vblock1 = (assemble(rbpos,:F,μ)+assemble(rbpos,:H,μ)+assemble(rbpos,:A_lift,μ)+
    assemble(rbpos,:M_lift,μ))
  vblock2 = assemble(rbpos,:B_lift,μ)

  rb_lhs = vcat(hcat(mblock11,mblock12),hcat(mblock21,mblock22))
  rb_rhs = vcat(vblock1,vblock2)
  rb_lhs,rb_rhs
end

function steady_navier_stokes_rb_system(
  rbpos::NTuple{N,RBParamOnlineStructure},
  μ::Param) where N

  lin_rb_lhs,lin_rb_rhs = steady_stokes_rb_system(rbpos,μ)
  navier_stokes_rb_system(rbpos,μ,lin_rb_lhs,lin_rb_rhs)
end

function unsteady_navier_stokes_rb_system(
  rbpos::NTuple{N,RBParamOnlineStructure},
  μ::Param) where N

  lin_rb_lhs,lin_rb_rhs = unsteady_stokes_rb_system(rbpos,μ)
  navier_stokes_rb_system(rbpos,μ,lin_rb_lhs,lin_rb_rhs)
end

function navier_stokes_rb_system(
  rbpos::NTuple{N,RBParamOnlineStructure},
  μ::Param,
  lin_rb_lhs::AbstractMatrix{Float},
  lin_rb_rhs::AbstractMatrix{Float}) where N

  rb_jac,rb_lhs = zeros(size(lin_rb_lhs)),zeros(size(lin_rb_lhs))
  rb_rhs = zeros(size(lin_rb_rhs))

  function rb_system!(
    Cun::AbstractMatrix{Float},
    Dun::AbstractMatrix{Float},
    Cliftun::AbstractMatrix{Float})

    rb_jac = lin_rb_lhs
    rb_jac[1:nu,1:nu] += Cun + Dun

    rb_lhs = lin_rb_lhs
    rb_lhs[1:nu,1:nu] += Cun

    rb_rhs = lin_rb_rhs
    rb_rhs[1:nu] += Cliftun
    #res_rb = lhs_rb*x_rb - rhs_rb

    rb_jac,rb_lhs,rb_rhs
  end

  function rb_system!(un)
    Cun = assemble(rbpos,:C,μ,un)
    Dun = assemble(rbpos,:D,μ,un)
    Cliftun = assemble(rbpos,:C_lift,μ,un)
    rb_system!(Cun,Dun,Cliftun)
  end

  rb_system!
end

function solve_rb_system(rb_lhs::Matrix{Float},rb_rhs::Matrix{Float})
  rb_lhs \ rb_rhs
end

function solve_rb_system(
  rb_system::Function,
  x0::AbstractVector{Float},
  args...;
  tol=1e-10,maxtol=1e10,maxit=20,kwargs...)

  #= function get_un_fun(ufe::Matrix{Float})
    unfe = compute_in_times(ufe,θ)
    n(tn) = findall(x -> x == tn,times)[1]
    tn -> FEFunction(Uk(tn),unfe[:,n(tn)])
  end =#

  x_rb = rb_initial_guess(x0,kwargs...)
  err = 1.
  iter = 0
  while norm(err) ≥ tol && iter < maxit

    if norm(err) ≥ maxtol
      printstyled("Newton iterations did not converge\n";color=:red)
      return x_rb
    end

    un_fun = get_un_fun(x_rb,args...)
    rb_jac,rb_lhs,rb_rhs = rb_system(un_fun)
    err = rb_jac \ Vector(rb_lhs*x_rb - rb_rhs)
    x_rb -= err
    iter += 1

    printstyled("Newton method: err = $(norm(err)), iter = $iter\n";color=:red)
  end

  x_rb
end

function get_un_fun(
  x_rb::Vector{Float},
  rb_space::RBSpaceSteady,
  U::ParamTrialFESpace,
  μ::Param)

  nstu = get_ns(rb_space)
  ufe = reconstruct_fe_sol(rb_space,x_rb[1:nstu])
  FEFunction(U,ufe,μ)
end

function get_un_fun(
  x_rb::Vector{Float},
  rb_space::RBSpaceUnsteady,
  U::ParamTransientTrialFESpace,
  μ::Param,
  time_info::TimeInfo)

  nstu = get_ns(rb_space)*get_nt(rb_space)
  ufe = reconstruct_fe_sol(rb_space,x_rb[1:nstu])
  unfe = compute_in_times(ufe,get_θ(time_info))
  FEFunction(U,unfe,μ,get_times(time_info))
end

function rb_initial_guess(
  rb_space::NTuple{2,RBSpace},
  args...)

  x0 = get_initial_guess(u,p,μ_offline,μk)
  rb_initial_guess(x0,rb_space,args...)
end

function rb_initial_guess(
  x0::AbstractArray,
  rb_space::NTuple{2,RBSpace},
  args...)

  Nu = get_Ns(rb_space[1])
  u0_rb = rb_spacetime_projection(rb_space[1],x0[1:Nu,:])
  p0_rb = rb_spacetime_projection(rb_space[2],x0[Nu+1:end,:])
  vcat(u0_rb,p0_rb)
end

function get_initial_guess(u::Snapshots,p::Snapshots,k::Int)
  nearest_neighbor(vcat(u,p),k)
end

function reconstruct_fe_sol(
  rb_space::RBSpaceSteady,
  rb_sol::AbstractArray{Float})

  id = get_id(rb_space)
  bs = get_basis_space(rb_space)
  Snapshots(id,bs*rb_sol,1,EMatrix{Float})
end

function reconstruct_fe_sol(
  rb_space::RBSpaceUnsteady,
  rb_sol::AbstractArray{Float})

  id = get_id(rb_space)
  bs = get_basis_space(rb_space)
  bt = get_basis_time(rb_space)
  ns = get_ns(rb_space)
  nt = get_nt(rb_space)

  rb_sol_resh = reshape(rb_sol,nt,ns)
  Snapshots(id,bs*(bt*rb_sol_resh)',1,EMatrix{Float})
end

function reconstruct_fe_sol(
  rb_space::NTuple{2,RBSpaceSteady},
  rb_sol::AbstractArray{Float})

  ns = get_ns.(rb_space)
  rb_sol_u,rb_sol_p = rb_sol[1:ns[1],:],rb_sol[1+ns[1]:end,:]
  reconstruct_fe_sol.(rb_space,(rb_sol_u,rb_sol_p))
end

function reconstruct_fe_sol(
  rb_space::NTuple{2,RBSpaceUnsteady},
  rb_sol::AbstractArray{Float})

  ns = get_ns.(rb_space)
  nt = get_nt.(rb_space)
  n = ns.*nt
  rb_sol_u,rb_sol_p = rb_sol[1:n[1],:],rb_sol[1+n[1]:end,:]
  reconstruct_fe_sol.(rb_space,(rb_sol_u,rb_sol_p))
end
