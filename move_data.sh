echo "Insert filename in nicholasmueller@MU00111074:~/git_repos/Mabla.jl/data : "
read filename 
scp -r nicholasmueller@MU00111074:~/git_repos/Mabla.jl/data/$filename ~/git_repos/Mabla.jl/data/$filename

