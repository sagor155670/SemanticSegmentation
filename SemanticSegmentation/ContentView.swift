//
//  ContentView.swift
//  SemanticSegmentation
//
//  Created by Saiful Islam Sagor on 11/3/24.
//

import SwiftUI
import VisionKit
import Vision

struct ContentView: View {
    
    @State var image:UIImage? = UIImage(named: "person3.jpg")!
    @State var maskImage:UIImage?
    var body: some View {
        VStack {
//            Image(uiImage: image!)
//                .resizable()
//                .imageScale(.large)
            ImageLiftView(imageName: "person3.avif")
                .frame(width: 700,height: 800)
                
            

        }
        .padding()
    }
}

#Preview {
    ContentView()
}
