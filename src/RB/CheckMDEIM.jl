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
  u1fun = FEFunction(V.test,u1)
  u1dfun = FEFunction(U.trial(μ1),u1)

  A,LA = assemble_matrix_and_lifting(opA)
  A1,LA1 = A(μ1),LA(μ1)
  _,LB = assemble_matrix_and_lifting(opB)
  LB1 = LB(μ1)
  C,LC = assemble_matrix_and_lifting(opC)
  C1,LC1 = C(u1fun),LC(u1dfun)
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
  errC1,errLC1 = C1rb - basisC*coeffC[1](u1fun),LC1rb - basisLC*coeffC[2](u1dfun)
  coeffD = compute_coefficient(rbopD,D_rb,μ1)
  errD1 = D1rb - basisD*coeffD[1](u1fun)

  norm(errA1),norm(errLA1),norm(errLB1),norm(errC1),norm(errLC1),norm(errD1)
end

function unsteady_navier_stokes()
  op = rbopC
  u1,μ1 = uh[1].snap,μ[1]
  u90,μ90 = uh[90].snap,μ[90]
  timesθ = get_timesθ(op)
  μ_mdeim = μ[1:info.mdeim_nsnap]
  findnz_map,snaps... = mdeim_snapshots(op,info,μ_mdeim,rbspace_uθ)
  bs,_,red_bs = space_quantities(snaps,findnz_map)
  btθ = get_basis_time(rbspace_uθ)
  bst = kron(btθ,bs)
  red_bst = kron(btθ,red_bs)

  function u(tθ)
    n = findall(x -> x == tθ,timesθ)[1]
    FEFunction(V.test,u1[:,n])
  end
  function ud(tθ)
    n = findall(x -> x == tθ,timesθ)[1]
    FEFunction(U.trial(μ1,tθ),u1[:,n])
  end
  function uK(tθ)
    n = findall(x -> x == tθ,timesθ)[1]
    FEFunction(V.test,u90[:,n])
  end
  function udK(tθ)
    n = findall(x -> x == tθ,timesθ)[1]
    FEFunction(U.trial(μ90,tθ),u90[:,n])
  end

  ########################### OK ###########################
  C,LC = assemble_matrix_and_lifting(op)
  C1 = Matrix([nonzero_values(C(u(tθ)),findnz_map) for tθ = timesθ])[:]
  C1rb = Matrix([bsu'*C(u(tθ))*bsu for tθ = timesθ])[:]
  errC = C1 - bst[1]*bst[1]'*C1
  errCrb = C1rb - red_bst[1]*red_bst[1]'*C1rb
  LC1 = Matrix([LC(ud(tθ)) for tθ = timesθ])[:]
  LC1rb = Matrix([bsu'*LC(ud(tθ)) for tθ = timesθ])[:]
  errLC = LC1 - bst[2]*bst[2]'*LC1
  errLCrb = LC1rb - red_bst[2]*red_bst[2]'*LC1rb
  norm(errC),norm(errCrb),norm(errLC),norm(errLCrb)
  ########################### OK ###########################
end

function unsteady_navier_stokes()
  op = rbopC
  u1,μ1 = uh[1].snap,μ[1]
  timesθ = get_timesθ(op)

  function u(tθ)
    n = findall(x -> x == tθ,timesθ)[1]
    FEFunction(V.test,u1[:,n])
  end
  function ud(tθ)
    n = findall(x -> x == tθ,timesθ)[1]
    FEFunction(U.trial(μ1,tθ),u1[:,n])
  end

  C,LC = assemble_matrix_and_lifting(op)

  bsu = rbspace[1].basis_space
  C1rb = Matrix([bsu'*C(u(tθ))*bsu for tθ = timesθ])[:]
  basisC = kron(C_rb[1].rbspace.basis_time,C_rb[1].rbspace.basis_space)
  coeffC = compute_coefficient(rbopC,C_rb,μ1)
  errC = C1rb - basisC*coeffC[1](u)[:]

  LC1rb = Matrix([bsu'*LC(ud(tθ)) for tθ = timesθ])[:]
  basisLC = kron(C_rb[2].rbspace.basis_time,C_rb[2].rbspace.basis_space)
  errLC = LC1rb - basisLC*coeffC[2](u)[:]

  norm(errC),norm(errLC)
end

function unsteady_navier_stokes()
  bsC = C_rb[1].rbspace.basis_space
  coeffC = compute_coefficient(rbopC,C_rb,μ1)
  c1θ = coeffC[1](u)[:]
  Nt = length(timesθ)
  err = Float[]
  for k = 1:Nt
    C11rb = (bsu'*C(u(timesθ[k]))*bsu)[:]
    errk = abs.(C11rb - bsC*c1θ[:,k])
    push!(err,maximum(errk))
  end
end

function unsteady_navier_stokes()
  u,μ = uh[90].snap[:,1],μ[90]
  timesθ = get_timesθ(opA)
  k = 60
  tθ = timesθ[k]
  ufun = FEFunction(V.test,u)
  udfun = FEFunction(U.trial(μ,tθ),u)

  function uall(tθ)
    n = findall(x -> x == tθ,timesθ)[1]
    FEFunction(V.test,uh[90].snap[:,n])
  end
  function udall(tθ)
    n = findall(x -> x == tθ,timesθ)[1]
    FEFunction(U.trial(μ,tθ),uh[90].snap[:,n])
  end

  A,LA = assemble_matrix_and_lifting(opA)
  A1,LA1 = A(μ,tθ),LA(μ,tθ)
  C,LC = assemble_matrix_and_lifting(opC)
  C1,LC1 = C(ufun),LC(udfun)
  D = assemble_matrix(opD)
  D1 = D(ufun)

  bsu = rbspace[1].basis_space
  btu = rbspace[1].basis_time
  bsp = rbspace[2].basis_space
  btp = rbspace[2].basis_time

  A1rb = bsu'*A1*bsu
  A1rb = Matrix(A1rb[:])
  C1rb = bsu'*C1*bsu
  C1rb = Matrix(C1rb[:])
  D1rb = bsu'*D1*bsu
  D1rb = Matrix(D1rb[:])
  LA1rb = bsu'*LA1
  LC1rb = bsu'*LC1

  Ainfo,_,_,Cinfo,Dinfo,_ = offinfo
  rbopA,A_rb = Ainfo
  rbopC,C_rb = Cinfo
  rbopD,D_rb = Dinfo

  basisA = A_rb[1].rbspace.basis_space
  basisC = C_rb[1].rbspace.basis_space
  basisD = D_rb[1].rbspace.basis_space
  basisLA = A_rb[2].rbspace.basis_space
  basisLC = C_rb[2].rbspace.basis_space

  coeffA = compute_coefficient(rbopA,A_rb,μ)
  errA1,errLA1 = A1rb - basisA*coeffA[1][k,:],LA1rb - basisLA*coeffA[2][k,:]
  coeffC = compute_coefficient(rbopC,C_rb,μ)
  errC1,errLC1 = C1rb - basisC*coeffC[1](uall)[k,:],LC1rb - basisLC*coeffC[2](udall)[k,:]
  coeffD = compute_coefficient(rbopD,D_rb,μ)
  errD1 = D1rb - basisD*coeffD[1](uall)[k,:]

  norm(errA1),norm(errLA1),norm(errC1),norm(errLC1),norm(errD1)
end

k = 90
function uall(tθ)
  n = findall(x -> x == tθ,timesθ)[1]
  FEFunction(V.test,uh[k].snap[:,n])
end
Con = online_assembler(Cinfo...,μ[k],false)[1]
Con90,Con90_shift = Con[1](uall),Con[2](uall)
C90(tθ) = assemble_matrix(rbopC)(uall(tθ))
C90rb = Matrix(Matrix([(bsu'*C90(tθ)*bsu)[:] for tθ=timesθ])')
myCon,myCon_shift = coeff_by_time_bases_bilin(rbopC,C90rb)
mm = zeros(size(Con90))
mm_shift = zeros(size(Con90_shift))
for i = 1:24
  for j = 1:24
    mm[1+(i-1)*13:i*13,1+(j-1)*13:j*13] = myCon[(i-1)*24+j]
    mm_shift[1+(i-1)*13:i*13,1+(j-1)*13:j*13] = myCon_shift[(i-1)*24+j]
  end
end

#= Aon = online_assembler(Ainfo...,μ[k],false)
Aon90 = Aon[1][1]
A90(t) = assemble_matrix(rbopA,t)(μ[k])
A90rb = Matrix(Matrix([(bsu'*A90(tθ)*bsu)[:] for tθ=timesθ])')
myAon,_ = coeff_by_time_bases_bilin(rbopA,A90rb)
mm = zeros(size(Aon90))
for i = 1:24
  for j = 1:24
    mm[1+(i-1)*13:i*13,1+(j-1)*13:j*13] = myAon[(i-1)*24+j]
  end
end =#
