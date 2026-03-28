plugins {
    id("com.android.application")
    // START: FlutterFire Configuration
    id("com.google.gms.google-services")
    // END: FlutterFire Configuration
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.hejitech.snowglobedemo"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_17.toString()
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "com.hejitech.snowglobedemo"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName

        ndk {
            // Include libc++_shared.so in the bundle
            // This is required for the Rust library to load correctly
            // @ts-ignore
            abiFilters.addAll(listOf("arm64-v8a", "x86_64"))
        }
    }

    flavorDimensions.add("version")
    productFlavors {
        create("standard") {
            dimension = "version"
            // No extra features (CPU only)
        }
        create("highPerf") {
            dimension = "version"
            // Enable GPU (Vulkan/OpenCL)
        }
        create("full") {
            dimension = "version"
            // Enable GPU + NPU (QNN/Hexagon)
        }
    }

    applicationVariants.all {
        val variant = this
        val flavor = variant.flavorName
        val rustFeatures = when (flavor) {
            "highPerf" -> "high_perf"
            "full" -> "high_perf,qnn"
            else -> "" // standard
        }
        
        // Pass features to all subprojects (including snowglobe_openai)
        project.rootProject.allprojects.forEach {
            it.extensions.extraProperties.set("CARGOKIT_RUST_FEATURES", rustFeatures)
        }
    }

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")
        }
    }
}

flutter {
    source = "../.."
}
