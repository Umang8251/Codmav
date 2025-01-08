CREATE DATABASE codmav1;
USE codmav1;

CREATE TABLE users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL
);

CREATE TABLE user_preferences (
    preference_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    height FLOAT NOT NULL,
    weight FLOAT NOT NULL,
    gender ENUM('Male', 'Female') NOT NULL,
    allergies VARCHAR(255) NOT NULL,
    diet_preference ENUM('veg', 'eggetarian', 'non-veg') NOT NULL,
    region VARCHAR(50) NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
        ON DELETE CASCADE ON UPDATE CASCADE
);

select * from users;


