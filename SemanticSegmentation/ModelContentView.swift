//
//  ModelContentView.swift
//  SemanticSegmentation
//
//  Created by Saiful Islam Sagor on 14/3/24.
//

import Foundation
import SwiftUI
import CoreML
import Vision
import CoreImage
struct ModelContentView: View {
    
    @State var image:UIImage = UIImage(named: "object4.avif")!
    @State var maskImage:UIImage?
//    var classifier = Classifier()
    @State var results: String?
    var body: some View {
        VStack {
                Image(uiImage: image)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: 300,height: 350)
            Button{
                guard let ciImage = CIImage(image: image) else{
                    print("can not convert UIImage to CIImage")
                    return
                }
                self.results =  detect(ciImage: ciImage)
            }label: {
                Text("predict")
                    .font(.title)
                    .fontWeight(.heavy)
                    .fontDesign(.serif)
            }
            Group {
                if let imageClass = self.results {
                    HStack{
                        Text("Image Categories: ")
                            .font(.caption)
                            .fontDesign(.serif)
                            .fontWeight(.heavy)
                        Text(imageClass)
                            .font(.caption)
                            .fontDesign(.serif)
                            .fontWeight(.heavy)

                    }
                }else{
                    HStack{
                        Text("Image Categories: N/A")
                            .font(.caption)
                            .fontDesign(.serif)
                            .fontWeight(.heavy)
                    }
                }
            }
        }
        .padding()
    }
    func detect(ciImage:CIImage) -> String {
       guard let model = try? VNCoreMLModel(for: MobileNetV2(configuration: MLModelConfiguration()).model) else{
           return ""
       }
       let request =  VNCoreMLRequest(model: model)
       let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
       try? handler.perform([request])
       
       guard let results =  request.results as? [VNClassificationObservation] else {
           return ""
       }
//       if let firstResult =  results.first{
//           return firstResult.identifier
//       }
        return results.first?.identifier ?? "No class"
   }
}
