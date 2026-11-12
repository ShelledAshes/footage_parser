# from profile import Profile 
from video_analyser import VideoAnalyser
from vprofile import Profile


# def create_profile():
#     return Profile()

def main():
    # create_profile()
    video_id = 1


    # video_path = "video/reference.mp4"
    # detector_type = "person"

    # video_analyser = VideoAnalyser(video_id, video_path, detector_type)
    # video_analyser.save_profile_screenshots()

    # print("profiles data", video_analyser.get_profiles_data())
    # print("get_profile_data_for_recognition", video_analyser.get_profile_data_for_recognition())


    profile_id = 1
    profile_path = f"{video_id}/profile_{1}"

    profile = Profile(id=profile_id)
    profile.load_from_json(profile_path)
    profile.save_video_clips(path=profile_path, padding_seconds=3)

    # video_analyser.save_profile_clips(1) # saving for profile id 1





if __name__ == "__main__":
    main()