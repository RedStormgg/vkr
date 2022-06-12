def getUserDirPath(username, email):
    return "userfiles/" + username + "_" + email.replace("@", "_")