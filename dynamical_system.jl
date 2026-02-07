"""
`solve_system(f,u0,n)`
solves the dynamical system 
``u_{n+1}=f(u_n)``
for N time steps, returns the solution at step `n` with parameters `p`.
"""
function solve_system(f,u0,p,n)
    u = u0
    for i in 1:n-1
        u = f(u,p)
    end
    return u
end
f(u,p) = u^2 - p*u
solve_system(f,1.25,0.25,100)
solve_system(f,1,0.25,1000)
function lorenz(u,p)
    α,σ,ρ,β=p
    du1 = u[1] + α*(σ*(u[2]-u[1]))
    du2 = u[2] + α*(u[1]*(ρ-u[3]) - u[2])
    du3 = u[3] + α*(u[1]*u[2] - β*u[3])
    [du1,du2,du3]
end
p = (0.02,10.0,28.0,8/3)
solve_system(lorenz,[1.0,0.0,0.0],p,1000)
u0 = [1.0,0.0,0.0]

function solve_system_save(f,u0,p,n)
    u = Vector{typeof(u0)}(undef,n)
    u[1] = u0
    for i in 1:n-1
        u[i+1] = f(u[i],p)
    end
    u
end
to_plot = solve_system_save(lorenz,u0,p,1000)
x = [to_plot[i][1] for i in 1:length(to_plot)]
y = [to_plot[i][2] for i in 1:length(to_plot)]
z = [to_plot[i][3] for i in 1:length(to_plot)]
plot(x,y,z)
