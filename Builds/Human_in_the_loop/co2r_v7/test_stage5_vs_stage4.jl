include("co2r_core.jl"); include("fitted_params.jl"); using Printf
N_CL=N_CL_DEFAULT; N_GDL=N_GDL_DEFAULT
cl=make_cl_mesh(N_CL); gdl=make_gdl_mesh(N_GDL); gg=build_gas_geom(gdl,cl)
ly4=make_layout(4,N_CL,N_GDL); ly5=Stage5Layout(N_CL,N_GDL); c_eq,_=equilibrium_vector()
F4=joinpath(@__DIR__,"output_fit","cache"); F5=joinpath(@__DIR__,"output_stage5","cache")
println("Stage-4-at-fitted-kinetics  vs  Stage 5 : where does the extra i_CO come from?")
@printf("%-8s %-5s %-9s %-9s %-9s %-9s %-9s %-9s %-9s\n",
        "V_RHE","stg","pH(1)","c_CO3(1)","c_Cs(1)","H_eff(1)","cCO2aq(1)","eta_CO2R","i_CO")
for V in (-1.200,-1.650,-2.075,-2.425)
  for (tag,dir,pre,ly) in (("S4",F4,"fit",ly4), ("S5",F5,"s5",ly5))
    u=load_solution(dir,pre,V); c=unpack_log_conc(u,N_CL)
    ps = ly.stage==5 ? u[phi_s_cl_idx5(ly,1)] : V-V_RHE_to_SHE_offset
    pl = ly.stage==5 ? u[phi_L_idx5(ly,1)]    : 0.0
    pH1=-log10(max(c[1,1]/1000,1e-30))
    eta = (ps-pl) - (U_CO_0 - nernst_fac*pH1)
    iCO,iH2,_=compute_currents(u,V,cl,ly)
    @printf("%+.4f  %-5s %8.3f %9.1f %9.1f %9.3f %9.4f %+9.4f %9.2f\n",
            V, tag, pH1, c[4,1], cCs_from_electroneutrality(view(c,:,1)),
            sechenov_H_CO2(view(c,:,1)), c[5,1], eta, iCO/10)
  end
end
println("\nSpatially averaged CL state (S4 -> S5):")
for V in (-1.650,-2.425)
  u4=load_solution(F4,"fit",V); u5=load_solution(F5,"s5",V)
  c4=unpack_log_conc(u4,N_CL); c5=unpack_log_conc(u5,N_CL)
  m(c,i)=sum(c[i,ix]*cl.dx[ix] for ix in 1:N_CL)/L_CL
  @printf("  V=%+.3f  <c_CO3> %8.1f -> %8.1f   <c_OH> %8.2f -> %8.2f   <c_CO2aq> %7.4f -> %7.4f   <pH> %.3f -> %.3f\n",
     V, m(c4,4), m(c5,4), m(c4,2), m(c5,2), m(c4,5), m(c5,5),
     -log10(m(c4,1)/1000), -log10(m(c5,1)/1000))
end
# where is the big relative i_CO step?
using DelimitedFiles
S=readdlm(joinpath(@__DIR__,"output_stage5","summary.csv"),',',skipstart=1)
V=S[:,1]; iCOv=S[:,7]
d=abs.(diff(iCOv))./max.(iCOv[1:end-1],1e-30)
k=argmax(d)
@printf("\nlargest relative i_CO step: V %+.3f -> %+.3f, i_CO %.3e -> %.3e mA/cm2 (%.1f%%)\n",
        V[k],V[k+1],iCOv[k],iCOv[k+1],100*d[k])
sel=findall(iCOv[1:end-1] .> 1.0)
@printf("largest relative i_CO step where i_CO > 1 mA/cm2: %.3f %% (at V=%+.3f)\n",
        100*maximum(d[sel]), V[sel[argmax(d[sel])]])
