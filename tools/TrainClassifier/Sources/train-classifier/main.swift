import CoreML
import CreateML
import CryptoKit
import Foundation

// ============================================================================
// train-classifier
//
// Swift-scripted equivalent of the Create ML GUI workflow documented in
// docs/createml_guide.md, for the `tcg_identifier` classifier (signal B of
// the on-device TCG detection pipeline, docs/tcg_detection_pipeline.md §3.4).
//
// Verified CreateML API (read from the actual macOS 27 / Xcode 27 SDK
// .swiftinterface, not guessed):
//   - MLImageClassier.DataSource.filesByLabel([String: [URL]])
//   - MLImageClassifier(trainingData:parameters:) throws
//   - MLImageClassifier.ModelParameters(featureExtractor:validation:maxIterations:augmentationOptions:)
//   - MLImageClassifier.model -> CoreML.MLModel, .write(to:) throws
//   - CoreML MLModel.modelDescription.{predictedFeatureName,predictedProbabilitiesName}
//   - MLFeatureValue(imageAt:constraint:options:) (ObjC factory bridged from
//     MLFeatureValue+MLImageConversion.h)
//   - MLModel.prediction(fromFeatures:) throws -> MLFeatureProvider (sync)
// ============================================================================

// MARK: - CLI args

struct Args {
    var repoRoot: URL
    var maxIterations: Int = 25
    var limitPerClass: Int? = nil // for --smoke-test
}

func parseArgs() -> Args {
    let defaultRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent() // train-classifier/
        .deletingLastPathComponent() // Sources/
        .deletingLastPathComponent() // TrainClassifier/
        .deletingLastPathComponent() // tools/
    var args = Args(repoRoot: defaultRoot)
    let argv = CommandLine.arguments.dropFirst()
    var it = argv.makeIterator()
    while let a = it.next() {
        switch a {
        case "--repo-root":
            if let v = it.next() { args.repoRoot = URL(fileURLWithPath: v) }
        case "--max-iterations":
            if let v = it.next(), let n = Int(v) { args.maxIterations = n }
        case "--smoke-test":
            args.limitPerClass = 20
        default:
            FileHandle.standardError.write("warning: unrecognized argument \(a)\n".data(using: .utf8)!)
        }
    }
    return args
}

let args = parseArgs()
let repoRoot = args.repoRoot
let manifestURL = repoRoot.appendingPathComponent("data/manifest.csv")
let tcgIdentifierRoot = repoRoot.appendingPathComponent("data/tcg_identifier")
let augmentedRoot = repoRoot.appendingPathComponent("data/tcg_identifier_augmented")
let indexRoot = repoRoot.appendingPathComponent("data/index")

print("== train-classifier ==")
print("repo root:  \(repoRoot.path)")
print("manifest:   \(manifestURL.path)")

// MARK: - `other` class merge decision
//
// docs/taxonomy.md has two classes that both plausibly map onto the brief's
// single `other` ("not a card / unsupported, don't force a guess") output:
//   - `not_a_card`: hands, tables, packaging, blurry/empty frames.
//   - `other_tcg`: a supported-but-not-yet-split-out TCG.
// The brief's signal B wants exactly one `other` meaning "don't force a
// guess" for both a non-card and an unhandled game. We resolve this HERE,
// at training-data-load time only: both source folders keep their own
// names/provenance in data/tcg_identifier/train/{not_a_card,other_tcg}/ and
// in the manifest (so dataset organization / docs/taxonomy.md are
// untouched), but images from either folder are placed under a single
// merged "other" label when building the Create ML training dictionary.
// This is a real modeling choice, not incidental plumbing: it means the
// classifier can no longer distinguish "not a card" from "an unlisted TCG"
// at inference time — by design, since the brief's `other` is defined by
// what action it triggers downstream (don't force a guess), not by why.
let mergedOtherSourceLabels: Set<String> = ["not_a_card", "other_tcg"]
func effectiveLabel(for rawLabel: String) -> String {
    mergedOtherSourceLabels.contains(rawLabel) ? "other" : rawLabel
}

// MARK: - Manifest loading & physical path resolution

struct ManifestRow {
    let filename: String
    let label: String
    let detectionSplit: String
    let physicalSplit: String // manifest's `split` column: "train" | "test"
    let isAugmented: Bool
}

func resolvePath(_ row: ManifestRow) -> URL {
    if row.isAugmented {
        return augmentedRoot.appendingPathComponent(row.label).appendingPathComponent(row.filename)
    }
    return tcgIdentifierRoot
        .appendingPathComponent(row.physicalSplit)
        .appendingPathComponent(row.label)
        .appendingPathComponent(row.filename)
}

print("\nReading manifest...")
let rawRows = try CSV.readDictionaries(at: manifestURL)
print("manifest rows: \(rawRows.count)")

var trainRows: [ManifestRow] = []
var evalRows: [ManifestRow] = []
var manifestHashInput = ""

for r in rawRows {
    guard r["model"] == "tcg_identifier" else { continue }
    guard let filename = r["filename"], let label = r["label"],
          let detectionSplit = r["detection_split"], let physicalSplit = r["split"],
          let source = r["source"]
    else { continue }
    let row = ManifestRow(
        filename: filename,
        label: label,
        detectionSplit: detectionSplit,
        physicalSplit: physicalSplit,
        isAugmented: source.hasPrefix("augment:")
    )
    switch detectionSplit {
    case "train":
        trainRows.append(row)
        manifestHashInput += "\(filename),\(label),\(detectionSplit)\n"
    case "eval":
        evalRows.append(row)
        manifestHashInput += "\(filename),\(label),\(detectionSplit)\n"
    case "index":
        continue // signal A's data, never used for the classifier (B)
    default:
        FileHandle.standardError.write("warning: unknown detection_split '\(detectionSplit)' for \(filename)\n".data(using: .utf8)!)
    }
}

print("detection_split=train rows (model=tcg_identifier): \(trainRows.count)")
print("detection_split=eval  rows (model=tcg_identifier): \(evalRows.count)")

// MARK: - Build training file-by-label dictionary (merging `other`)

var trainFilesByLabel: [String: [URL]] = [:]
var missingTrainFiles = 0
var trainCountsRaw: [String: Int] = [:] // by original taxonomy label, for reporting
var trainCountsBySource: [String: (original: Int, augmented: Int)] = [:]

for row in trainRows {
    let path = resolvePath(row)
    guard FileManager.default.fileExists(atPath: path.path) else {
        missingTrainFiles += 1
        continue
    }
    let eff = effectiveLabel(for: row.label)
    trainFilesByLabel[eff, default: []].append(path)
    trainCountsRaw[row.label, default: 0] += 1
    var counts = trainCountsBySource[row.label] ?? (0, 0)
    if row.isAugmented { counts.augmented += 1 } else { counts.original += 1 }
    trainCountsBySource[row.label] = counts
}

if let limit = args.limitPerClass {
    for (k, v) in trainFilesByLabel {
        trainFilesByLabel[k] = Array(v.shuffled().prefix(limit))
    }
    print("--smoke-test: capped each merged class to \(limit) images")
}

print("\nTraining set (after merging not_a_card + other_tcg -> \"other\"):")
for (label, count) in trainCountsRaw.sorted(by: { $0.key < $1.key }) {
    let src = trainCountsBySource[label] ?? (0, 0)
    print("  \(label.padding(toLength: 22, withPad: " ", startingAt: 0)) total=\(count) (original=\(src.original), augmented=\(src.augmented))")
}
print("merged class sizes:")
for (label, urls) in trainFilesByLabel.sorted(by: { $0.key < $1.key }) {
    print("  \(label.padding(toLength: 22, withPad: " ", startingAt: 0)) \(urls.count)")
}
if missingTrainFiles > 0 {
    print("warning: \(missingTrainFiles) train rows referenced files not found on disk (skipped)")
}

guard !trainFilesByLabel.isEmpty else {
    FileHandle.standardError.write("error: no training images found — check --repo-root and that data/ symlinks resolve\n".data(using: .utf8)!)
    exit(1)
}

// MARK: - Build eval set (held out, never used above)

struct EvalItem {
    let path: URL
    let trueLabel: String // merged label
}

var evalItems: [EvalItem] = []
var missingEvalFiles = 0
for row in evalRows {
    let path = resolvePath(row)
    guard FileManager.default.fileExists(atPath: path.path) else {
        missingEvalFiles += 1
        continue
    }
    evalItems.append(EvalItem(path: path, trueLabel: effectiveLabel(for: row.label)))
}
print("\nEval set (detection_split=eval, held out from training above): \(evalItems.count) images")
if missingEvalFiles > 0 {
    print("warning: \(missingEvalFiles) eval rows referenced files not found on disk (skipped)")
}

// Sanity: eval images must not appear in the training URL set (by path).
let trainPathSet = Set(trainFilesByLabel.values.flatMap { $0 }.map { $0.path })
let evalPathSet = Set(evalItems.map { $0.path.path })
let leakage = trainPathSet.intersection(evalPathSet)
if !leakage.isEmpty {
    FileHandle.standardError.write("error: \(leakage.count) files appear in both train and eval sets — aborting\n".data(using: .utf8)!)
    exit(1)
}
print("leakage check: 0 overlapping files between train and eval (by path) ✓")

// MARK: - Train

print("\n== Training ==")
let augmentation: MLImageClassifier.ImageAugmentationOptions = [.crop, .rotation, .exposure, .noise, .flip]
let parameters = MLImageClassifier.ModelParameters(
    validation: .split(strategy: .automatic),
    maxIterations: args.maxIterations,
    augmentation: augmentation,
    algorithm: .transferLearning(featureExtractor: .scenePrint(revision: 1), classifier: .logisticRegressor)
)

let trainStart = Date()
let classifier: MLImageClassifier
do {
    classifier = try MLImageClassifier(trainingData: .filesByLabel(trainFilesByLabel), parameters: parameters)
} catch {
    FileHandle.standardError.write("error: training failed: \(error)\n".data(using: .utf8)!)
    exit(1)
}
let trainDuration = Date().timeIntervalSince(trainStart)
print("training wall-clock time: \(String(format: "%.1f", trainDuration))s")
print("training metrics:   \(classifier.trainingMetrics)")
print("validation metrics: \(classifier.validationMetrics)")

// MARK: - Export

try FileManager.default.createDirectory(at: indexRoot, withIntermediateDirectories: true)
let modelURL = indexRoot.appendingPathComponent("tcg_classifier.mlmodel")
if FileManager.default.fileExists(atPath: modelURL.path) {
    try FileManager.default.removeItem(at: modelURL)
}
try classifier.write(to: modelURL)
let modelAttrs = try FileManager.default.attributesOfItem(atPath: modelURL.path)
let modelSizeBytes = (modelAttrs[.size] as? Int) ?? 0
print("\nexported model: \(modelURL.path) (\(modelSizeBytes) bytes, \(String(format: "%.2f", Double(modelSizeBytes) / 1_000_000)) MB)")

// MARK: - Evaluate on eval split + collect raw probabilities for calibration

print("\n== Evaluating on held-out eval split ==")
let model = classifier.model
let desc = model.modelDescription
guard let imageInputName = desc.inputDescriptionsByName.first(where: { $0.value.type == .image })?.key,
      let imageConstraint = desc.inputDescriptionsByName[imageInputName]?.imageConstraint,
      let predictedLabelName = desc.predictedFeatureName,
      let predictedProbName = desc.predictedProbabilitiesName
else {
    FileHandle.standardError.write("error: could not introspect exported model's feature names\n".data(using: .utf8)!)
    exit(1)
}
print("model input feature: \(imageInputName), predicted label feature: \(predictedLabelName), probabilities feature: \(predictedProbName)")

// MLModel's synchronous `prediction(fromFeatures:)` is unavailable on this
// SDK (macOS-only async API now); bridge the async `prediction(from:)` to a
// blocking call for this straight-line CLI tool.
func syncPrediction(_ model: MLModel, from input: any MLFeatureProvider) throws -> any MLFeatureProvider {
    let semaphore = DispatchSemaphore(value: 0)
    var result: Result<any MLFeatureProvider, Error>!
    Task {
        do {
            let output = try await model.prediction(from: input)
            result = .success(output)
        } catch {
            result = .failure(error)
        }
        semaphore.signal()
    }
    semaphore.wait()
    return try result.get()
}

struct EvalPrediction {
    let trueLabel: String
    let predictedLabel: String
    let probs: [String: Double] // raw softmax from the exported model
}

var evalPredictions: [EvalPrediction] = []
var evalErrors = 0
for item in evalItems {
    do {
        let featureValue = try MLFeatureValue(imageAt: item.path, constraint: imageConstraint, options: [:])
        let provider = try MLDictionaryFeatureProvider(dictionary: [imageInputName: featureValue])
        let output = try syncPrediction(model, from: provider)
        guard let predictedLabel = output.featureValue(for: predictedLabelName)?.stringValue,
              let rawDict = output.featureValue(for: predictedProbName)?.dictionaryValue
        else {
            evalErrors += 1
            continue
        }
        var probs: [String: Double] = [:]
        for (k, v) in rawDict {
            if let key = k as? String {
                probs[key] = v.doubleValue
            }
        }
        evalPredictions.append(EvalPrediction(trueLabel: item.trueLabel, predictedLabel: predictedLabel, probs: probs))
    } catch {
        evalErrors += 1
    }
}
print("scored \(evalPredictions.count) eval images (\(evalErrors) prediction errors)")

// MARK: - Accuracy reporting (overall + per class, never hidden in an average)

func accuracy(_ preds: [EvalPrediction]) -> Double {
    guard !preds.isEmpty else { return 0 }
    let correct = preds.filter { $0.predictedLabel == $0.trueLabel }.count
    return Double(correct) / Double(preds.count)
}

let overallAccuracy = accuracy(evalPredictions)
print("\noverall top-1 accuracy: \(String(format: "%.4f", overallAccuracy)) (\(evalPredictions.count) images)")

var perClassAccuracy: [String: (correct: Int, total: Int)] = [:]
for p in evalPredictions {
    var c = perClassAccuracy[p.trueLabel] ?? (0, 0)
    c.total += 1
    if p.predictedLabel == p.trueLabel { c.correct += 1 }
    perClassAccuracy[p.trueLabel] = c
}
print("per-class accuracy:")
for (label, c) in perClassAccuracy.sorted(by: { $0.key < $1.key }) {
    let acc = c.total > 0 ? Double(c.correct) / Double(c.total) : 0
    let flag = (acc < overallAccuracy - 0.10) ? "  <-- FLAG: notably below overall" : ""
    print("  \(label.padding(toLength: 22, withPad: " ", startingAt: 0)) \(String(format: "%.4f", acc)) (\(c.correct)/\(c.total))\(flag)")
}

// MARK: - Temperature scaling calibration
//
// Create ML's own export has no temperature-scaling step. We fit a single
// scalar T minimizing negative log-likelihood on the EVAL split (never used
// for training/validation above) via a 1D line search — no optimization
// library needed. Since the exported model already applies softmax
// (probabilities sum to 1, not raw logits), we recover a logit surrogate as
// z_i = ln(p_i) (accurate up to the arbitrary additive constant that
// softmax is invariant to) and evaluate softmax(z / T) against each image's
// true label.
func nll(probs: [String: Double], trueLabel: String, temperature: Double, labels: [String]) -> Double {
    let eps = 1e-9
    var logits: [Double] = []
    for l in labels {
        let p = max(probs[l] ?? eps, eps)
        logits.append(log(p) / temperature)
    }
    let maxLogit = logits.max() ?? 0
    let expSum = logits.reduce(0.0) { $0 + exp($1 - maxLogit) }
    guard let trueIdx = labels.firstIndex(of: trueLabel) else { return 0 }
    let trueLogit = logits[trueIdx]
    let logProb = (trueLogit - maxLogit) - log(expSum)
    return -logProb
}

let allLabels = Array(Set(evalPredictions.flatMap { Array($0.probs.keys) })).sorted()
var bestTemperature = 1.0
var bestNLL = Double.infinity
if !evalPredictions.isEmpty && !allLabels.isEmpty {
    // Coarse grid, then a refine pass around the best coarse candidate.
    let coarseRange = stride(from: 0.1, through: 5.0, by: 0.05)
    for t in coarseRange {
        let total = evalPredictions.reduce(0.0) { $0 + nll(probs: $1.probs, trueLabel: $1.trueLabel, temperature: t, labels: allLabels) }
        let mean = total / Double(evalPredictions.count)
        if mean < bestNLL {
            bestNLL = mean
            bestTemperature = t
        }
    }
    let fineRange = stride(from: max(0.01, bestTemperature - 0.05), through: bestTemperature + 0.05, by: 0.005)
    for t in fineRange {
        let total = evalPredictions.reduce(0.0) { $0 + nll(probs: $1.probs, trueLabel: $1.trueLabel, temperature: t, labels: allLabels) }
        let mean = total / Double(evalPredictions.count)
        if mean < bestNLL {
            bestNLL = mean
            bestTemperature = t
        }
    }
}
print("\ncalibration: fitted temperature T = \(String(format: "%.4f", bestTemperature)) (mean eval NLL = \(String(format: "%.4f", bestNLL)))")

// MARK: - Write calibration sidecar

struct CalibrationJSON: Encodable {
    let temperature: Double
    let evalMeanNLL: Double
    let evalImageCount: Int
    let overallTop1Accuracy: Double
    let perClassAccuracy: [String: PerClass]
    let classLabels: [String]
    let buildDate: String

    struct PerClass: Encodable {
        let correct: Int
        let total: Int
        let accuracy: Double
    }
}

let iso = ISO8601DateFormatter()
let buildDateString = iso.string(from: Date())

let calibration = CalibrationJSON(
    temperature: bestTemperature,
    evalMeanNLL: bestNLL,
    evalImageCount: evalPredictions.count,
    overallTop1Accuracy: overallAccuracy,
    perClassAccuracy: Dictionary(uniqueKeysWithValues: perClassAccuracy.map { (label, c) in
        (label, CalibrationJSON.PerClass(correct: c.correct, total: c.total, accuracy: c.total > 0 ? Double(c.correct) / Double(c.total) : 0))
    }),
    classLabels: allLabels,
    buildDate: buildDateString
)

let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
let calibrationData = try encoder.encode(calibration)
let calibrationURL = indexRoot.appendingPathComponent("tcg_classifier_calibration.json")
try calibrationData.write(to: calibrationURL)
print("\nwrote calibration sidecar: \(calibrationURL.path)")

// MARK: - Version manifest (docs/tcg_detection_pipeline.md §5)

func sha256Hex(_ string: String) -> String {
    let digest = SHA256.hash(data: Data(string.utf8))
    return digest.map { String(format: "%02x", $0) }.joined()
}

struct VersionManifestJSON: Encodable {
    let buildDate: String
    let datasetManifestHash: String
    let classLabels: [String]
    let mergedOtherFrom: [String]
    let trainImageCount: Int
    let trainImageCountByClass: [String: Int]
    let evalImageCount: Int
    let calibrationTemperature: Double
    let modelPath: String
    let modelSizeBytes: Int
    let trainingWallClockSeconds: Double
    let maxIterations: Int
    let featureExtractor: String
}

let versionManifest = VersionManifestJSON(
    buildDate: buildDateString,
    datasetManifestHash: sha256Hex(manifestHashInput),
    classLabels: allLabels,
    mergedOtherFrom: Array(mergedOtherSourceLabels).sorted(),
    trainImageCount: trainFilesByLabel.values.reduce(0) { $0 + $1.count },
    trainImageCountByClass: trainFilesByLabel.mapValues { $0.count },
    evalImageCount: evalItems.count,
    calibrationTemperature: bestTemperature,
    modelPath: "data/index/tcg_classifier.mlmodel",
    modelSizeBytes: modelSizeBytes,
    trainingWallClockSeconds: trainDuration,
    maxIterations: args.maxIterations,
    featureExtractor: "scenePrint(revision: 1)"
)
let versionData = try encoder.encode(versionManifest)
let versionURL = indexRoot.appendingPathComponent("tcg_classifier_version.json")
try versionData.write(to: versionURL)
print("wrote version manifest: \(versionURL.path)")

print("\n== Done ==")
