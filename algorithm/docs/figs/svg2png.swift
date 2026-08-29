#!/usr/bin/env swift
import Foundation
import WebKit
import AppKit

let svgPath = CommandLine.arguments[1]
let outPath = CommandLine.arguments[2]
let W: CGFloat = 1920, H: CGFloat = 1080

let app = NSApplication.shared
app.setActivationPolicy(.accessory)

let cfg = WKWebViewConfiguration()
let web = WKWebView(frame: NSRect(x: 0, y: 0, width: W, height: H), configuration: cfg)

let svg = try! String(contentsOfFile: svgPath, encoding: .utf8)
let html = """
<html><head><meta charset="utf-8">
<style>html,body{margin:0;padding:0;background:#fff;width:1920px;height:1080px;overflow:hidden}
svg{display:block;width:1920px;height:1080px}</style>
</head><body>\(svg)</body></html>
"""

final class Nav: NSObject, WKNavigationDelegate {
    let web: WKWebView; let out: String
    init(web: WKWebView, out: String) { self.web = web; self.out = out }
    func webView(_ w: WKWebView, didFinish n: WKNavigation!) {
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            let c = WKSnapshotConfiguration()
            c.rect = CGRect(x: 0, y: 0, width: W, height: H)
            w.takeSnapshot(with: c) { img, err in
                guard let img = img,
                      let tiff = img.tiffRepresentation,
                      let rep = NSBitmapImageRep(data: tiff) else {
                    FileHandle.standardError.write("snapshot failed: \(String(describing: err))\n".data(using: .utf8)!)
                    exit(1)
                }
                let scaled = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 1920, pixelsHigh: 1080,
                    bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                    colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
                NSGraphicsContext.saveGraphicsState()
                NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: scaled)
                rep.draw(in: NSRect(x: 0, y: 0, width: 1920, height: 1080))
                NSGraphicsContext.restoreGraphicsState()
                let png = scaled.representation(using: .png, properties: [:])!
                try! png.write(to: URL(fileURLWithPath: self.out))
                print("ok \(self.out)")
                exit(0)
            }
        }
    }
}

let nav = Nav(web: web, out: outPath)
web.navigationDelegate = nav
web.loadHTMLString(html, baseURL: URL(fileURLWithPath: svgPath).deletingLastPathComponent())
app.run()
