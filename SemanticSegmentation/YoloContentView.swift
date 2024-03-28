//
//  YoloContentView.swift
//  SemanticSegmentation
//
//  Created by Saiful Islam Sagor on 14/3/24.
//

import Foundation
import SwiftUI
import Vision
import CoreML

struct YoloContentView: View {
    @State var selectedImageUrl: URL? = nil
    @State var isShowingPicker:Bool = false
    @State private var image: UIImage = UIImage(named: "object2.avif")!
    @State private var rectangles: [RectangleData] = []
    
    var body: some View {
        VStack {
            Image(uiImage: image)
                .resizable()
                .aspectRatio(contentMode: .fit)
//                .frame(width: 350,height: 400)
                .overlay(RectangleOverlay(rectangles: $rectangles))
            
            Button {
                self.isShowingPicker = true
            }label: {
                Text("Select image")
                    .font(.title)
                    .fontWeight(.heavy)
                    .fontDesign(.serif)
            }
            Button {
                if self.image != nil {
                    detectObjects()
//                     generateMask()
                }else{
                    Text("Select a image first! ")
                        .font(.caption)
                        .fontDesign(.serif)
                        .fontWeight(.heavy)
                }
            } label: {
                Text("Detect Objects")
                    .font(.title)
                    .fontWeight(.heavy)
                    .fontDesign(.serif)
            }

        }.sheet(isPresented: $isShowingPicker, content: {
            MediaPicker(selectedMediaUrl: $selectedImageUrl, isShowingPicker: $isShowingPicker, image: $image, mediaTypes: ["public.image"])
        })

    }
    
    func detectObjects() {
        guard let model = try? VNCoreMLModel(for:yolov8x(configuration: MLModelConfiguration()).model) else {
            print("Failed to load model")
            return
        }
        
        let request = VNCoreMLRequest(model: model) { request, error in
            if let error = error {
                print("Failed to process request: \(error)")
                return
            }
            
            guard let results = request.results as? [VNRecognizedObjectObservation] else {
                print("Failed to obtain results")
                return
            }
            
            DispatchQueue.main.async {
                self.rectangles = results.map { RectangleData(rect: $0.boundingBox) }
            }
        }
        
        let handler = VNImageRequestHandler(cgImage: self.image.cgImage!, options: [:])
        try? handler.perform([request])
    }
//    

    

    
    func getMaskProtosFromOutput(
        output: MLMultiArray,
        rows: Int,
        columns: Int,
        tubes: Int
    ) -> [[UInt8]] {
        var masks: [[UInt8]] = []
        for tube in 0..<tubes {
            var mask: [UInt8] = []
            for i in 0..<(rows*columns) {
                let index = tube*(rows*columns)+i
                mask.append(UInt8(truncating: output[index]))
            }
            masks.append(mask)
        }
        return masks
    }
    
    private func crop(
        mask: [Float],
        maskSize: (width: Int, height: Int),
        box: XYXY
    ) -> [Float] {
        let rows = maskSize.height
        let columns = maskSize.width
        
        let x1 = Int(box.x1 / 4)
        let y1 = Int(box.y1 / 4)
        let x2 = Int(box.x2 / 4)
        let y2 = Int(box.y2 / 4)
        
        var croppedArr: [Float] = []
        for row in 0..<rows {
            for column in 0..<columns {
                if column >= x1 && column <= x2 && row >= y1 && row <= y2 {
                    croppedArr.append(mask[row*columns+column])
                } else {
                    croppedArr.append(0)
                }
            }
        }
        return croppedArr
    }
    
    func getPredictionsFromOutput(
        output: MLMultiArray,
        rows: Int,
        columns: Int,
        numberOfClasses: Int,
        inputImgSize: CGSize
    ) -> [Prediction] {
        guard output.count != 0 else {
            return []
        }
        var predictions = [Prediction]()
        for i in 0..<columns {
            let centerX = Float(truncating: output[0*columns+i])
            let centerY = Float(truncating: output[1*columns+i])
            let width   = Float(truncating: output[2*columns+i])
            let height  = Float(truncating: output[3*columns+i])
            
            let (classIndex, score) = {
                var classIndex: Int = 0
                var heighestScore: Float = 0
                for j in 0..<numberOfClasses {
                    let score = Float(truncating: output[(4+j)*columns+i])
                    if score > heighestScore {
                        heighestScore = score
                        classIndex = j
                    }
                }
                return (classIndex, heighestScore)
            }()
            
            let maskCoefficients = {
                var coefficients: [Float] = []
                for k in 0..<32 {
                    coefficients.append(Float(truncating: output[(4+numberOfClasses+k)*columns+i]))
                }
                return coefficients
            }()
            
            // Convert box from xywh to xyxy
            let left = centerX - width/2
            let top = centerY - height/2
            let right = centerX + width/2
            let bottom = centerY + height/2
            
            let prediction = Prediction(
                classIndex: classIndex,
                score: score,
                xyxy: (left, top, right, bottom),
                maskCoefficients: maskCoefficients,
                inputImgSize: inputImgSize
            )
            predictions.append(prediction)
        }
        
        return predictions
    }
    
}
func nonMaximumSuppression(
    predictions: [Prediction],
    iouThreshold: Float,
    limit: Int
) -> [Prediction] {
    guard !predictions.isEmpty else {
        return []
    }
    
    let sortedIndices = predictions.indices.sorted {
        predictions[$0].score > predictions[$1].score
    }
    
    var selected: [Prediction] = []
    var active = [Bool](repeating: true, count: predictions.count)
    var numActive = active.count

    // The algorithm is simple: Start with the box that has the highest score.
    // Remove any remaining boxes that overlap it more than the given threshold
    // amount. If there are any boxes left (i.e. these did not overlap with any
    // previous boxes), then repeat this procedure, until no more boxes remain
    // or the limit has been reached.
    outer: for i in 0..<predictions.count {
        
        if active[i] {
            
            let boxA = predictions[sortedIndices[i]]
            selected.append(boxA)
            
            if selected.count >= limit { break }

            for j in i+1..<predictions.count {
            
                if active[j] {
            
                    let boxB = predictions[sortedIndices[j]]
                    
                    if IOU(a: boxA.xyxy, b: boxB.xyxy) > iouThreshold {
                        
                        active[j] = false
                        numActive -= 1
                       
                        if numActive <= 0 { break outer }
                    
                    }
                
                }
            
            }
        }
        
    }
    return selected
}

private func IOU(a: XYXY, b: XYXY) -> Float {
       // Calculate the intersection coordinates
       let x1 = max(a.x1, b.x1)
       let y1 = max(a.y1, b.y1)
       let x2 = max(a.x2, b.x2)
       let y2 = max(a.y1, b.y2)
       
       // Calculate the intersection area
       let intersection = max(x2 - x1, 0) * max(y2 - y1, 0)
       
       // Calculate the union area
       let area1 = (a.x2 - a.x1) * (a.y2 - a.y1)
       let area2 = (b.x2 - b.x1) * (b.y2 - b.y1)
       let union = area1 + area2 - intersection
       
       // Calculate the IoU score
       let iou = intersection / union
       
       return iou
   }

struct RectangleOverlay: View {
    @Binding var rectangles: [RectangleData]
    
    var body: some View {
        GeometryReader { geometry in
            ForEach(rectangles) { data in
                Rectangle()
                    .path(in: CGRect(x: data.rect.minX * geometry.size.width, y: (1 - data.rect.maxY) * geometry.size.height, width: data.rect.width * geometry.size.width, height: data.rect.height * geometry.size.height))
                    .stroke(Color.red)
                
            }
            
        }
    }
}

struct RectangleData: Identifiable {
    let id = UUID()
    let rect: CGRect
}

typealias XYXY = (x1: Float, y1: Float, x2: Float, y2: Float)

struct Prediction {
    let id = UUID()
    
    let classIndex: Int
    let score: Float
    let xyxy: XYXY
    let maskCoefficients: [Float]
    
    let inputImgSize: CGSize
}

func getPixelBuffer(from image: UIImage) -> CVPixelBuffer? {
    // Convert UIImage to CGImage
    guard let cgImage = image.cgImage else {
        return nil
    }
    
    // Create options for pixel buffer attributes
    let options: [String: Any] = [
        kCVPixelBufferCGImageCompatibilityKey as String: true,
        kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
    ]
    
    var pixelBuffer: CVPixelBuffer?
    let width = cgImage.width
    let height = cgImage.height
    
    // Create pixel buffer
    let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                     width,
                                     height,
                                     kCVPixelFormatType_32ARGB,
                                     options as CFDictionary,
                                     &pixelBuffer)
    
    guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
        return nil
    }
    
    // Lock the base address of the pixel buffer
    CVPixelBufferLockBaseAddress(buffer, [])
    
    // Get the base address of the pixel buffer
    guard let pixelData = CVPixelBufferGetBaseAddress(buffer) else {
        return nil
    }
    
    // Create a bitmap context
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue)
    guard let context = CGContext(data: pixelData,
                                  width: width,
                                  height: height,
                                  bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: bitmapInfo.rawValue) else {
        return nil
    }
    
    // Draw the image onto the bitmap context
    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
    
    // Unlock the pixel buffer
    CVPixelBufferUnlockBaseAddress(buffer, [])
    
    return buffer
}


func resizeImage(_ image: UIImage, to newSize: CGSize) -> UIImage? {
    UIGraphicsBeginImageContextWithOptions(newSize, false, 0.0)
    image.draw(in: CGRect(origin: CGPoint.zero, size: newSize))
    let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()
    return resizedImage
}
