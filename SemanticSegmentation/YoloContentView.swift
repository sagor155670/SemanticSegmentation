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
        guard let model = try? VNCoreMLModel(for:yolov8l(configuration: MLModelConfiguration()).model) else {
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
