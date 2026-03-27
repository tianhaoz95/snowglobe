#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint rust_lib_snowglobe_openai.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'rust_lib_snowglobe_openai'
  s.version          = '0.0.1'
  s.summary          = 'A new Flutter FFI plugin project.'
  s.description      = <<-DESC
A new Flutter FFI plugin project.
                       DESC
  s.homepage         = 'http://example.com'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Your Company' => 'email@example.com' }

  # This will ensure the source files in Classes/ are included in the native
  # builds of apps using this FFI plugin. Podspec does not support relative
  # paths, so Classes contains a forwarder C file that relatively imports
  # `../src/*` so that the C sources can be shared among all target platforms.
    s.source           = { :path => '.' }
    s.source_files     = 'Classes/**/*'
    s.dependency 'FlutterMacOS'
  
    s.platform = :osx, '13.0'
    s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES' }
    s.swift_version = '5.0'
  
    s.frameworks = 'SystemConfiguration', 'CoreServices', 'Security', 'Metal', 'MetalPerformanceShaders', 'MetalPerformanceShadersGraph'
    s.libraries = 'c++'
  
    # Check if prebuilt XCFramework exists
    has_prebuilt = File.exist?(File.expand_path('rust_lib_snowglobe_openai.xcframework', __dir__))

    if has_prebuilt
      s.vendored_frameworks = 'rust_lib_snowglobe_openai.xcframework'
      s.pod_target_xcconfig = {
        'DEFINES_MODULE' => 'YES',
        'ARCHS[sdk=macosx*]' => 'arm64',
        'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386'
      }
    else
      s.script_phase = {
        :name => 'Build Rust library',
        :script => 'sh "$PODS_TARGET_SRCROOT/../cargokit/build_pod.sh" ../rust rust_lib_snowglobe_openai',
        :execution_position => :before_compile,
        :input_files => ['${BUILT_PRODUCTS_DIR}/cargokit_phony'],
        :output_files => ["${BUILT_PRODUCTS_DIR}/librust_lib_snowglobe_openai.a"],
      }
      s.pod_target_xcconfig = {
        'DEFINES_MODULE' => 'YES',
        'ARCHS[sdk=macosx*]' => 'arm64',
        'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
        'OTHER_LDFLAGS' => '-force_load ${BUILT_PRODUCTS_DIR}/librust_lib_snowglobe_openai.a',
      }
    end
  end
  