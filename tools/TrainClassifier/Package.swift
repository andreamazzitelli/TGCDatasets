// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "TrainClassifier",
    platforms: [
        .macOS(.v13)
    ],
    targets: [
        .executableTarget(
            name: "train-classifier",
            path: "Sources/train-classifier"
        )
    ]
)
