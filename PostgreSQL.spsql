-- Create the 'users' table
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY, -- SERIAL is used for auto-increment in PostgreSQL
    username VARCHAR(50) NOT NULL
);

-- Create the 'user_preferences' table
CREATE TABLE user_preferences (
    preference_id SERIAL PRIMARY KEY, -- SERIAL is used for auto-increment in PostgreSQL
    user_id INT NOT NULL,
    height FLOAT NOT NULL,
    weight FLOAT NOT NULL,
    gender VARCHAR(10) CHECK (gender IN ('Male', 'Female')) NOT NULL, -- ENUM replaced with CHECK constraint
    allergies VARCHAR(255) NOT NULL,
    diet_preference VARCHAR(20) CHECK (diet_preference IN ('veg', 'eggetarian', 'non-veg')) NOT NULL, -- ENUM replaced with CHECK constraint
    region VARCHAR(50) NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
        ON DELETE CASCADE ON UPDATE CASCADE
);
