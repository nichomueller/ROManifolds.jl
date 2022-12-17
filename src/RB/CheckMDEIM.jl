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

function steady_stokes()
  μ1 = μ[1]
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
  coeffB = compute_coefficient(rbopB,B_rb,μ1)
  errB1,errLB1 = B1rb - basisB*coeffB[1],LB1rb - basisLB*coeffB[2]
end
