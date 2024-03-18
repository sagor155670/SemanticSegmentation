//
//  Classifier.swift
//  SemanticSegmentation
//
//  Created by Saiful Islam Sagor on 14/3/24.
//

import Foundation
import CoreML
import Vision
import CoreImage

class Classifier{
     var results: String?
    
     func detect(ciImage:CIImage) -> String {
        guard let model = try? VNCoreMLModel(for: YOLOv3(configuration: MLModelConfiguration()).model) else{
            return ""
        }
        let request =  VNCoreMLRequest(model: model)
        let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
        try? handler.perform([request])
        
        guard let results =  request.results as? [VNClassificationObservation] else {
            return ""
        }
        if let firstResult =  results.first{
            return firstResult.identifier
        }
         return ""
    }
}
