function norm_test(matrb,basis)
  coeff = basis \ matrb
  err = matrb - basis*coeff
  norm(err)
end

function norm_test_app(rbop,mdeim,matrb,basis)
  coeff = compute_coefficient(rbop,mdeim,μ1)
  err = matrb - basis*coeff
  norm(err)
end

function unsteady_poisson()
  Nt = get_Nt(opA)
  μ1 = μ[1]
  A,LA = assemble_matrix_and_lifting(opA)
  A1,LA1 = A(μ1),LA(μ1)
  M,LM = assemble_matrix_and_lifting(opM)
  M1,LM1 = M(μ1),LM(μ1)
  F1 = assemble_vector(opF)(μ1)
  H1 = assemble_vector(opH)(μ1)

  bsu = rbspace.basis_space
  btu = rbspace.basis_time

  A1rb = [bsu'*A1[k]*bsu for k = 1:Nt]
  A1rb = Matrix([A1rb[k][:] for k = 1:Nt])
  M1rb = bsu'*M1[1]*bsu
  F1rb = bsu'*F1
  H1rb = bsu'*H1
  LA1rb = bsu'*LA1
  LM1rb = bsu'*LM1

  basisA = A_rb.rbspace.basis_space
  basisF = F_rb.rbspace.basis_space
  basisH = H_rb.rbspace.basis_space
  basisLA = A_rb_lift.rbspace.basis_space
  basisLM = M_rb_lift.rbspace.basis_space

  norm_test(A1rb,basisA)
  norm_test(LA1rb,basisLA)
  norm_test(LM1rb,basisLM)

  norm_test_app(rbopA,A_rb,A1rb,basisA)
  norm_test_app(rbopF,F_rb,F1rb,basisF)
  norm_test_app(rbopH,H_rb,H1rb,basisH)
  norm_test_app(rbopA_lift,A_rb_lift,LA1rb,basisLA)
  norm_test_app(rbopM_lift,M_rb_lift,LM1rb,basisLM)

  lhs = assemble_rb_system(rbopA,A_rb,μ1),assemble_rb_system(rbopM,M_rb,μ1)
  rhs = assemble_rb_system(rbopF,F_rb,μ1),assemble_rb_system(rbopH,H_rb,μ1)
  lift = assemble_rb_system(rbopA_lift,A_rb_lift,μ1),assemble_rb_system(rbopM_lift,M_rb_lift,μ1)

  ns,nt = get_ns(get_rbspace_row(rbopA)),get_nt(get_rbspace_row(rbopA))
  A1rb_s = reshape(A1rb,ns,ns,Nt)
  M1rb_s = repeat(M1rb,1,1,Nt)
  F1rb_s = reshape(F1rb,ns,1,Nt)
  H1rb_s = reshape(H1rb,ns,1,Nt)
  LA1rb_s = reshape(LA1rb,ns,1,Nt)
  LM1rb_s = reshape(LM1rb,ns,1,Nt)

  A1rb_st = zeros(ns*nt,ns*nt)
  M1rb_st = zeros(ns*nt,ns*nt)
  F1rb_st = zeros(ns*nt,1)
  H1rb_st = zeros(ns*nt,1)
  LA1rb_st = zeros(ns*nt,1)
  LM1rb_st = zeros(ns*nt,1)

  for k = 1:Nt
    for is = 1:ns
      for it = 1:nt
        ist = (is-1)*nt+it
        F1rb_st[ist,1] += F1rb_s[is,1,k]*btu[k,it]
        H1rb_st[ist,1] += H1rb_s[is,1,k]*btu[k,it]
        LA1rb_st[ist,1] += LA1rb_s[is,1,k]*btu[k,it]
        LM1rb_st[ist,1] += LM1rb_s[is,1,k]*btu[k,it]
        for js = 1:ns
          for jt = 1:nt
            jst = (js-1)*nt+jt
            A1rb_st[ist,jst] += A1rb_s[is,js,k]*btu[k,it]*btu[k,jt]
            M1rb_st[ist,jst] += M1rb_s[is,js,k]*btu[k,it]*btu[k,jt]
          end
        end
      end
    end
  end

  dtθ = get_dt(opA)*get_θ(opA)

  norm(lhs[1][1]-A1rb_st)
  norm(lhs[2][1]*dtθ-M1rb_st)
  norm(rhs[1]-F1rb_st)
  norm(rhs[2]-H1rb_st)
  norm(lift[1]-LA1rb_st)
  norm(lift[2]-LM1rb_st)

  u1 = uh[1].snap
  Π = get_basis_spacetime(rbspace)
  uhat1 = Π'*u1[:]
end

function steady_stokes(μ1::Param)
  A,LA = assemble_matrix_and_lifting(opA)
  A1,LA1 = A(μ1),LA(μ1)
  B,LB = assemble_matrix_and_lifting(opB)
  B1,LB1 = B(μ1),LB(μ1)

  bsu = rbspace[1].basis_space
  bsp = rbspace[2].basis_space

  A1rb = bsu'*A1*bsu
  A1rb = Matrix(A1rb[:])
  LA1rb = bsu'*LA1
  LB1rb = bsp'*LB1

  basisA = A_rb[1].rbspace.basis_space
  basisLA = A_rb[2].rbspace.basis_space
  basisLB = B_rb[2].rbspace.basis_space

  norm_test(A1rb,basisA)
  norm_test(LA1rb,basisLA)
  norm_test(LB1rb,basisLB)

  coeffA = compute_coefficient(rbopA,A_rb,μ1)
  errA1,errLA1 = A1rb - basisA*coeffA[1],LA1rb - basisLA*coeffA[2]
  opB_lift = RBLiftingOperator(rbopB)
  coeffLB1 = compute_coefficient(opB_lift,B_rb[2],μ1)
  errLB1 = LB1rb - basisLB*coeffLB1

  norm(errA1),norm(errLA1),norm(errLB1)
end

function steady_navier_stokes()
  u1,μ1 = uh.snap[:,1],μ[1]
  u1fun = FEFunction(V.test,u1)#FEFunction(U.trial(μ1),u1)

  A,LA = assemble_matrix_and_lifting(opA)
  A1,LA1 = A(μ1),LA(μ1)
  _,LB = assemble_matrix_and_lifting(opB)
  LB1 = LB(μ1)
  C,LC = assemble_matrix_and_lifting(opC)
  C1,LC1 = C(u1fun),LC(u1fun)
  D = assemble_matrix(opD)
  D1 = D(u1fun)

  bsu = rbspace[1].basis_space
  bsp = rbspace[2].basis_space

  A1rb = bsu'*A1*bsu
  A1rb = Matrix(A1rb[:])
  C1rb = bsu'*C1*bsu
  C1rb = Matrix(C1rb[:])
  D1rb = bsu'*D1*bsu
  D1rb = Matrix(D1rb[:])
  LA1rb = bsu'*LA1
  LB1rb = bsp'*LB1
  LC1rb = bsu'*LC1

  basisA = A_rb[1].rbspace.basis_space
  basisC = C_rb[1].rbspace.basis_space
  basisD = D_rb[1].rbspace.basis_space
  basisLA = A_rb[2].rbspace.basis_space
  basisLB = B_rb[2].rbspace.basis_space
  basisLC = C_rb[2].rbspace.basis_space

  #norm_test(C1rb,basisC)
  #norm_test(D1rb,basisD)

  coeffA = compute_coefficient(rbopA,A_rb,μ1)
  errA1,errLA1 = A1rb - basisA*coeffA[1],LA1rb - basisLA*coeffA[2]
  opB_lift = RBLiftingOperator(rbopB)
  coeffLB1 = compute_coefficient(opB_lift,B_rb[2],μ1)
  errLB1 = LB1rb - basisLB*coeffLB1
  coeffC = compute_coefficient(rbopC,C_rb,μ1)
  errC1,errLC1 = C1rb - basisC*coeffC[1](u1fun),LC1rb - basisLC*coeffC[2](u1fun)
  coeffD = compute_coefficient(rbopD,D_rb,μ1)
  errD1 = D1rb - basisD*coeffD[1](u1fun)

  norm(errA1),norm(errLA1),norm(errLB1),norm(errC1),norm(errLC1),norm(errD1)
end

function steady_navier_stokes1()
  u1,μ1 = uh.snap[:,1],μ[1]
  u1fun = FEFunction(U.trial(μ1),u1)

  op = rbopC
  id = get_id(op)
  bfun,bfun_lift = basis_as_fefun(op)
  findnz_map = get_findnz_map(op,bfun_lift(μ1,1))
  C = assemble_matrix(op)

  function snapshot(k::Int,n::Int)
    println("Nonlinear lift snapshot number $((k-1)*ns+n), $id")
    b = bfun_lift(μ[k],n)
    v = nonzero_values(C(b),findnz_map)
    v
  end
  ns = size(get_basis_space_col(op),2)
  nparam = 2
  vals = Vector{Float}[]
  for k = 1:nparam
    for n = 1:ns
      push!(vals,snapshot(k,n))
    end
  end
  snaps = Snapshots(id,vals)
  rbspaceC = mdeim_basis(info,snaps)

  C1 = Vector(C(u1fun)[:][findnz_map])
  basis_C = rbspaceC.basis_space
  errC = C1-basis_C*basis_C'*C1
  norm(errC)
end

function spacetime_mdeim1()
  μ_mdeim = μ[1:info.mdeim_nsnap]
  snaps = mdeim_snapshots(op,info,μ_mdeim)[1]
  rbspace = mdeim_basis(info,snaps)
  Π = kron(rbspace.basis_time,rbspace.basis_space)
  idx = mdeim_idx(Π)
  redΠ = Π[idx,:]
  A1 = snaps[1].snap[:]
  θ = redΠ \ A1[idx]
  A1mdeim = Π * θ
  norm(A1-A1mdeim) # small

  #= also true:
  idx_space = mdeim_idx(rbspace.basis_space)
  idx_time = mdeim_idx(rbspace.basis_time)
  idx_st = zeros(Int,216)
  Qs,Qt,Nz = 36,6,68800
  for it=1:Qt
    idx_st[(it-1)*Qs+1:it*Qs] = idx_space .+ (idx_time[it]-1)*Nz
  end
  @assert idx == idx_st
  redΠ = kron(rbspace.basis_time[idx_time,:],rbspace.basis_space[idx_space,:])
  @assert isapprox(Π[idx,:],redΠ)
  A1 = snaps[1].snap[:]
  θ = redΠ \ A1[idx]
  A1mdeim = Π * θ
  norm(A1-A1mdeim) # small =#
end

function important()
  op = rbopA
  μ_mdeim = μ[1:info.mdeim_nsnap]
  snaps = mdeim_snapshots(op,info,μ_mdeim)[1]
  rbspace = mdeim_basis(info,snaps)

  # traditional method
  info = RBInfoUnsteady(ptype,mesh,root;ϵ=1e-5,nsnap=80,mdeim_snap=20,load_offline=false,
    save_offline=false,st_mdeim=false)
  idx = mdeim_idx(rbspace)
  red_lu_factors = get_red_lu_factors(info,rbspace,idx)
  idx = recast_in_full_dim(op,idx)
  #red_meas = get_red_measure(op,idx,measures,:dΩ)
  timesθ = get_timesθ(op)
  A = assemble_red_structure(op,measures.dΩ,μ[1],idx[1],timesθ)
  θ_s = solve_lu(A,red_lu_factors)
  btbtc = coeff_by_time_bases_bilin(op,θ_s)[1]

  # space-time method
  info = RBInfoUnsteady(ptype,mesh,root;ϵ=1e-5,nsnap=80,mdeim_snap=20,load_offline=false,
    save_offline=false,st_mdeim=true)
  red_rbspace = project_mdeim_basis(info,op,rbspace)[1]
  idx = mdeim_idx(rbspace)
  red_lu_factors = get_red_lu_factors(info,rbspace,idx)
  idx_space = mdeim_idx(rbspace.basis_space)
  idx_time = mdeim_idx(rbspace.basis_time)
  idx_st = zeros(Int,216)
  Qs,Qt,Nz = 36,6,68800
  for it=1:Qt
    idx_st[(it-1)*Qs+1:it*Qs] = idx_space .+ (idx_time[it]-1)*Nz
  end
  redΠ = kron(rbspace.basis_time[idx_time,:],rbspace.basis_space[idx_space,:])
  A1 = snaps[1].snap[:]
  θ_st = redΠ \ A1[idx_st]
  bt_full = get_basis_time(rbspace)
  Nt,Qt = size(bt_full)
  rbrow = get_rbspace_row(op)
  rbcol = get_rbspace_col(op)
  brow = get_basis_time(rbrow)
  bcol = get_basis_time(rbcol)
  nrow = size(brow,2)
  ncol = size(bcol,2)
  idx = 1:Nt
  time_proj_fun(it,jt,q) = sum(brow[idx,it].*bcol[idx,jt].*bt_full[idx,q])
  time_proj_fun(jt,q) = Broadcasting(it -> time_proj_fun(it,jt,q))(1:nrow)
  time_proj_fun(q) = Matrix(Broadcasting(jt -> time_proj_fun(jt,q))(1:ncol))
  #= time_proj_fun(it,jt,q) = sum(brow[idx,it].*bcol[idx,jt].*bt_full[idx,q])
  time_proj_fun(it,q) = Broadcasting(jt -> time_proj_fun(it,jt,q))(1:ncol)
  time_proj_fun(q) = Matrix(Broadcasting(it -> time_proj_fun(it,q))(1:nrow)) =#
  bt_block = time_proj_fun.(1:Qt)
  my_idx(qs) = [(i-1)*Qs+qs for i=1:Qt]
  function trythis(q)
    qt = 1 + Int.(floor.((q-1)/Qs))
    qs = q - (qt-1)*Qs
    sum(bt_block[qt] .* θ_st[my_idx(qs)])
  end
  yee1 = Matrix([trythis(q) for q=1:Qs*Qt])

  ############### YYYYYYYYYYYYYYYYEAAAAAAAAAAAAAHHHHHHHHHHHHHH #################
  dog(qs) = bt_full*θ_st[my_idx(qs)]
  cane = Matrix([dog(qs) for qs=1:Qs])
  isapprox(cane,θ_s)
  ############### YYYYYYYYYYYYYYYYEAAAAAAAAAAAAAHHHHHHHHHHHHHH #################
  temp(it,jt,qs) = sum(brow[idx,it].*bcol[idx,jt].*dog(qs)[idx])
  temp(jt,qs) = Broadcasting(it -> temp(it,jt,qs))(1:nrow)
  temp(qs) = Matrix(Broadcasting(jt -> temp(jt,qs))(1:ncol))
  v(qs) = maximum(abs.(temp(qs)-btbtc[qs]))
  yee = v.(1:Qs)
end

function compare()
  μ_mdeim = μ[1:info.mdeim_nsnap]
  snaps = mdeim_snapshots(op,info,μ_mdeim)[1]
  rbspace = mdeim_basis(info,snaps)
  info = RBInfoUnsteady(ptype,mesh,root;ϵ=1e-5,nsnap=80,mdeim_snap=20,load_offline=false,
    save_offline=false,st_mdeim=true)
  red_rbspace = project_mdeim_basis(op,rbspace)[1]
  idx = mdeim_idx(rbspace)
  red_lu_factors = get_red_lu_factors(info,rbspace,idx)
  idx_space = mdeim_idx(rbspace.basis_space)
  idx_time = mdeim_idx(rbspace.basis_time)
  idx = recast_in_full_dim(op,(idx_space,idx_time))

  idx_space,idx_time = get_idx_space(mdeim),get_idx_time(mdeim)
end
