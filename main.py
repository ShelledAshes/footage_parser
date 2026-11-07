# from profile import Profile 
from video_analyser import VideoAnalyser


# def create_profile():
#     return Profile()

def main():
    # create_profile()
    video_path = "video/reference.mp4"
    detector_type = "person"

    video_analyser = VideoAnalyser(video_path, detector_type)
    video_analyser.save_profile_screenshots()




if __name__ == "__main__":
    main()