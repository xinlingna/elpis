# rm -r Release ; mkdir Release
# cd Release
# cmake -DCMAKE_BUILD_TYPE=Release .. &> logCMAKE.log
# make -j  
# echo "_Release_Mode Project! Please find the exec in release repository."

# cmake -DCMAKE_BUILD_TYPE=Release .. &> logCMAKE.log:
                                 # .. 表示 CMake 去读取上一级目录的 CMakeLists.txt 文件


rm -rf build
mkdir build
cd build

# cmake -G Ninja -DCMAKE_BUILD_TYPE=Release .. &> ../logCMAKE.log
cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON .. &> logCMAKE.log

ninja

echo "_Release_Mode Project! Build completed using Ninja!"

