import Foundation

/// Minimal RFC4180-ish CSV reader. `data/manifest.csv`'s `notes` column
/// routinely contains embedded commas inside double quotes (card names like
/// `"Suraya, Archangel of Erudition"`), so a naive `split(separator: ",")`
/// silently misaligns columns. This handles quoted fields and `""` escaped
/// quotes, which is all the manifest ever produces (single-line fields,
/// no embedded newlines observed in practice).
enum CSV {
    static func parseLine(_ line: Substring) -> [String] {
        var fields: [String] = []
        var current = ""
        var inQuotes = false
        let chars = Array(line)
        var i = 0
        while i < chars.count {
            let c = chars[i]
            if inQuotes {
                if c == "\"" {
                    if i + 1 < chars.count && chars[i + 1] == "\"" {
                        current.append("\"")
                        i += 1
                    } else {
                        inQuotes = false
                    }
                } else {
                    current.append(c)
                }
            } else {
                if c == "\"" {
                    inQuotes = true
                } else if c == "," {
                    fields.append(current)
                    current = ""
                } else {
                    current.append(c)
                }
            }
            i += 1
        }
        fields.append(current)
        return fields
    }

    /// Reads a CSV file into an array of [column: value] dictionaries, keyed
    /// by the header row. Rows with a field count mismatch are skipped with
    /// a warning (should not happen against a well-formed manifest, but this
    /// is better than crashing or silently misaligning columns).
    static func readDictionaries(at url: URL) throws -> [[String: String]] {
        let content = try String(contentsOf: url, encoding: .utf8)
        var lines = content.split(separator: "\n", omittingEmptySubsequences: true)
        guard !lines.isEmpty else { return [] }
        let header = parseLine(lines.removeFirst())
        var results: [[String: String]] = []
        results.reserveCapacity(lines.count)
        for (idx, line) in lines.enumerated() {
            let fields = parseLine(line)
            guard fields.count == header.count else {
                FileHandle.standardError.write(
                    "warning: manifest row \(idx + 2) has \(fields.count) fields, expected \(header.count); skipping\n"
                        .data(using: .utf8)!
                )
                continue
            }
            var dict: [String: String] = [:]
            for (h, v) in zip(header, fields) {
                dict[h] = v
            }
            results.append(dict)
        }
        return results
    }
}
