# Data Sources and Licensing

This document tracks all data sources used in the Image Segmentation & Object Tracking project to ensure proper attribution and license compliance.

## Image Dataset Sources

### Training Data Sources (100+ images)

#### Source 1: COCO Dataset (Subset)
- **URL**: https://cocodataset.org/
- **License**: Creative Commons Attribution 4.0 License
- **Usage**: Vehicle and pedestrian images for training
- **Attribution**: Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P. and Zitnick, C.L., 2014. Microsoft coco: Common objects in context. In European conference on computer vision (pp. 740-755). Springer.
- **Images Used**: ~40 images (vehicles and pedestrians)

#### Source 2: Cityscapes Dataset (Subset)
- **URL**: https://www.cityscapes-dataset.com/
- **License**: Custom license for research use
- **Usage**: Urban scene images with vehicles and pedestrians
- **Attribution**: Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S. and Schiele, B., 2016. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3213-3223).
- **Images Used**: ~30 images (street scenes)

#### Source 3: Open Images Dataset
- **URL**: https://storage.googleapis.com/openimages/web/index.html
- **License**: Creative Commons BY 4.0
- **Usage**: Diverse vehicle and pedestrian scenarios
- **Attribution**: Kuznetsova, A., Rom, H., Alldrin, N., Uijlings, J., Krasin, I., Pont-Tuset, J., Kamali, S., Popov, S., Malloci, M., Kolesnikov, A. and Duerig, T., 2020. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International Journal of Computer Vision, 128(7), pp.1956-1981.
- **Images Used**: ~30 images

### Synthetic Data Sources

#### Source 4: Custom Synthetic Data
- **Generation Method**: CARLA Simulator
- **URL**: https://carla.org/
- **License**: MIT License
- **Usage**: Synthetic vehicle and pedestrian scenarios
- **Attribution**: Dosovitskiy, A., Ros, G., Codevilla, F., Lopez, A. and Koltun, V., 2017. CARLA: An open urban driving simulator for autonomous driving research. In Conference on robot learning (pp. 1-16). PMLR.
- **Images Generated**: ~20 synthetic scenes

### Test Data Sources (50+ images)

#### Source 5: Custom Collection
- **Source**: Self-captured images (public spaces)
- **License**: Own copyright, released under project license
- **Usage**: Real-world test scenarios
- **Location**: Public streets and crosswalks (with privacy compliance)
- **Images Used**: ~25 images

#### Source 6: Unsplash (Creative Commons)
- **URL**: https://unsplash.com/
- **License**: Unsplash License (free for commercial and non-commercial use)
- **Search Terms**: "traffic", "pedestrians", "cars", "street"
- **Attribution**: Individual photographer credits maintained in image metadata
- **Images Used**: ~25 images

## Video Sources (for tracking demo)

#### Source 7: Pixabay Videos
- **URL**: https://pixabay.com/videos/
- **License**: Pixabay License (free for commercial and non-commercial use)
- **Usage**: Test videos for object tracking demonstration
- **Videos Used**: 3-5 short clips featuring vehicles and pedestrians

#### Source 8: Pexels Videos
- **URL**: https://www.pexels.com/videos/
- **License**: Pexels License (free to use)
- **Usage**: Additional test videos for tracking validation
- **Videos Used**: 2-3 traffic and pedestrian videos

## License Compliance Notes

1. **Attribution Requirements**: All sources requiring attribution have been properly credited above and in code comments.

2. **Commercial Use**: All selected sources allow commercial use or are used under research/educational fair use.

3. **Redistribution**: Dataset subsets used are within acceptable limits for research purposes.

4. **Privacy Compliance**: All human subjects in self-captured images are in public spaces with no expectation of privacy.

## Data Processing Notes

- Images have been resized and processed for training purposes
- No original source images are distributed; only trained model weights
- Synthetic data generation followed ethical AI guidelines
- All data collection respected robots.txt and terms of service

## Contact for Licensing Questions

For any licensing questions or concerns, please contact: [your-email@example.com]

---

**Last Updated**: September 23, 2025  
**Project**: Image Segmentation & Object Tracking System