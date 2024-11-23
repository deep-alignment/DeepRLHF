                                 
#if defined(__cplusplus)                                            
__NV_CPLUSPLUS __cplusplus                                          
#endif /* defined(__cplusplus) */                                   
#if defined(__GNUC__)                                               
__NV_GNUC__ __GNUC__                                                
#if defined(__GNUC_MINOR__)                                         
__NV_GNUC_MINOR__ __GNUC_MINOR__                                    
#endif /* defined(__GNUC_MINOR__) */                                
#if defined(__GNUC_PATCHLEVEL__)                                    
__NV_GNUC_PATCHLEVEL__ __GNUC_PATCHLEVEL__                          
#endif /* defined(__GNUC_PATCHLEVEL__) */                           
#endif /* defined(__GNUC__) */                                      
#if defined(__PGIC__)                                               
__NV_PGIC__ __PGIC__                                                
#if defined(__PGIC_MINOR__)                                         
__NV_PGIC_MINOR__ __PGIC_MINOR__                                    
#endif /* defined(__PGIC_MINOR__) */                                
#if defined(__PGIC_PATCHLEVEL__)                                    
__NV_PGIC_PATCHLEVEL__ __PGIC_PATCHLEVEL__                          
#endif /* defined(__PGIC_PATCHLEVEL__) */                           
#if defined(__PGLLVM__)                                             
__NV_PGLLVM__                                                       
#endif  /* defined(__PGLLVM__) */                                   
#endif /* defined(__PGIC__) */                                      
#if defined(__clang__)                                              
__NV_CLANG__                                                        
#if defined(__clang_major__)                                        
__NV_CLANG_MAJOR__ __clang_major__                                  
#endif /* defined(__clang_major__) */                               
#if defined(__clang_minor__)                                        
__NV_CLANG_MINOR__ __clang_minor__                                  
#endif /* defined(__clang_minor__) */                               
#if defined(__clang_patchlevel__)                                   
__NV_CLANG_PATCHLEVEL__ __clang_patchlevel__                        
#endif /* defined(__clang_patchlevel__) */                          
#if defined(__apple_build_version__)                                
__NV_APPLE_BUILD_VERSION__                                          
#endif /* defined(__apple_build_version__) */                       
#endif /* defined(__clang__) */                                     
#if defined(__PGI)                                                  
__NV_PGI__                                                          
#endif /* defined(__PGI) */                                         
#if defined(__ICC)                                                  
__NV_ICC__ __ICC                                                    
#endif /* defined(__ICC) */                                         
#if defined(__INTEL_LLVM_COMPILER )                                 
__NV_INTEL_LLVM_COMPILER__ __INTEL_LLVM_COMPILER                    
#endif /* defined(__ICC) */                                         
#if defined(__ibmxl__)                                              
__NV_XLC__                                                          
#endif /* defined(__ibmxl__) */                                     
#if defined(__gnu_linux__)                                          
__NV_LINUX__                                                        
#elif defined(__linux__) && !defined(__ANDROID__)                   
__NV_LINUX__                                                        
#endif /* defined(__gnu_linux__) */                                 
#if defined(__APPLE__) && defined(__MACH__)                         
__NV_MACOSX__                                                       
#endif /* defined(__APPLE__) && defined(__MACH__) */                
#if defined(__ANDROID__)                                            
__NV_ANDROID__                                                      
#endif /* defined(__ANDROID__) */                                   
#if defined(__QNX__)                                                
__NV_QNX__                                                          
#endif /* defined(__QNX__) */                                       
#if defined(__i686) || defined(__i386)                              
__NV_X86__                                                          
#endif /* defined(__i686) || defined(__i386) */                     
#if defined(__x86_64)                                               
__NV_X86_64__                                                       
#endif /* defined(__x86_64) */                                      
#if defined(__arm__)                                                
__NV_ARMv7__                                                        
#endif /* defined(__arm__) */                                       
#if defined(__aarch64__)                                            
__NV_AARCH64__                                                      
#endif /* defined(__aarch64__) */                                   
#if defined(__powerpc64__) && defined(__LITTLE_ENDIAN__)            
__NV_PPC64LE__                                                      
#endif /* defined(__powerpc64__) && defined(__LITTLE_ENDIAN__) */   
#if defined(__HORIZON__)                                            
__NV_HORIZON__                                                      
#endif /* defined(__HORIZON__) */                                   
#if defined(_MSC_VER)                                               
__NV_MSC_VER _MSC_VER                                               
#endif /* defined(_MS_VER) */                                       
#if defined(_MSC_FULL_VER)                                          
__NV_MSC_FULL_VER _MSC_FULL_VER                                     
#endif /* defined(_MSC_FULL_VER) */                                 
#if defined(__CHAR_UNSIGNED__) || defined(_CHAR_UNSIGNED)           
__NV_CHAR_UNSIGNED__                                                
#endif /* defined(__CHAR_UNSIGNED__) || defined(_CHAR_UNSIGNED) */  
#if defined(__WCHAR_UNSIGNED__) || defined(_WCHAR_UNSIGNED)         
__NV_WCHAR_UNSIGNED__                                               
#endif /* defined(__WCHAR_UNSIGNED__) || defined(_WCHAR_UNSIGNED) */
