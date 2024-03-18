//
//  MediaPicker.swift
//  SemanticSegmentation
//
//  Created by Saiful Islam Sagor on 14/3/24.
//

import Foundation
import UIKit
import SwiftUI
import Photos

struct MediaPicker: UIViewControllerRepresentable {
    @Binding var selectedMediaUrl: URL?
    @Binding var isShowingPicker: Bool
    @Binding var image: UIImage
    var mediaTypes: [String]

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let mediaPicker = UIImagePickerController()
        print(mediaPicker.description)
        mediaPicker.sourceType = .photoLibrary
        mediaPicker.mediaTypes = mediaTypes
        mediaPicker.videoQuality = .typeHigh // Set the video quality to high
        mediaPicker.videoExportPreset = AVAssetExportPresetPassthrough // Preserve original video quality
        mediaPicker.delegate = context.coordinator
        return mediaPicker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {
        // Update UI if needed
    }

    func makeCoordinator() -> Coordinator {
        return Coordinator(self)
    }
}

class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    var parent: MediaPicker

    init(_ picker: MediaPicker) {
        self.parent = picker
    }

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if let mediaURL = info[.mediaURL] as? URL {
            // Selected media is a video
            parent.selectedMediaUrl = mediaURL
        } else if let image = info[.originalImage] as? UIImage {
            // Selected media is an image
            parent.image = image
//            if let imageUrl = info[.imageURL] as? URL {
//                parent.selectedMediaUrl = imageUrl
//
//
//            } else {
//                if let asset = info[.phAsset] as? PHAsset {
//                    PHImageManager.default().requestImageDataAndOrientation(for: asset, options: nil) { data, _, _, info in
//                        if let imageUrl = info?["PHImageFileURLKey"] as? URL {
//                            DispatchQueue.main.async {
//                                self.parent.selectedMediaUrl = imageUrl
//                            }
//                        }
//                    }
//                }
//            }
        }
        parent.isShowingPicker = false
    }

    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        parent.isShowingPicker = false
    }
}

