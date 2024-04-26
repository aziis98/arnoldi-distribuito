# Progetto di Calcolo Scientifico

```bash shell
# Create a build directory
$ mkdir build
$ cd build

# Build, Compile, Run
$ cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
$ make
$ mpirun ./main

# Or inline for development
$ cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON .. && make && mpirun ./main
```

## VSCode

Install the `clangd` extension and put the following in `.vscode/settings.json`

```json
{
    "clangd.arguments": [
        "-background-index", 
        "-compile-commands-dir=build"
    ]
}
```