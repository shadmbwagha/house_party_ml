import joblib
import pandas as pd
from django.shortcuts import render
from django import forms
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load('house_party_ml/models/attendance_model.pkl')


# Define a form for collecting input data
class AttendanceForm(forms.Form):
    age = forms.IntegerField(label='Age')
    gender = forms.ChoiceField(choices=[('M', 'Male'), ('F', 'Female')], label='Gender')
    marital_status = forms.ChoiceField(choices=[('Single', 'Single'), ('Married', 'Married')], label='Marital Status')
    number_of_children = forms.IntegerField(label='Number of Children')
    department = forms.ChoiceField(
        choices=[('IT', 'IT'), ('HR', 'HR'), ('Finance', 'Finance'), ('Marketing', 'Marketing'), ('Sales', 'Sales')],
        label='Department')
    position = forms.ChoiceField(
        choices=[('Software Engineer', 'Software Engineer'), ('HR Manager', 'HR Manager'), ('Accountant', 'Accountant'),
                 ('Marketing Specialist', 'Marketing Specialist'), ('Sales Executive', 'Sales Executive')],
        label='Position')
    employment_status = forms.ChoiceField(
        choices=[('Full-time', 'Full-time'), ('Part-time', 'Part-time'), ('Contract', 'Contract')],
        label='Employment Status')
    shift_timing = forms.ChoiceField(choices=[('Day', 'Day'), ('Night', 'Night')], label='Shift Timing')
    length_of_service = forms.IntegerField(label='Length of Service')
    work_hours = forms.IntegerField(label='Work Hours')
    overtime_days = forms.IntegerField(label='Overtime Days')
    vacation_days = forms.IntegerField(label='Vacation Days')
    past_event_attendance = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')], label='Past Event Attendance')
    work_days_last_month = forms.IntegerField(label='Work Days Last Month')
    absent_days_last_month = forms.IntegerField(label='Absent Days Last Month')
    event_interest = forms.IntegerField(label='Event Interest')
    event_type_preference = forms.ChoiceField(
        choices=[('Social', 'Social'), ('Professional', 'Professional'), ('Family-friendly', 'Family-friendly')],
        label='Event Type Preference')
    event_timing_preference = forms.ChoiceField(
        choices=[('Weekday', 'Weekday'), ('Weekend', 'Weekend'), ('Evening', 'Evening')],
        label='Event Timing Preference')
    family_commitments = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')], label='Family Commitments')
    health_issues = forms.ChoiceField(choices=[('None', 'None'), ('Mild', 'Mild'), ('Severe', 'Severe')],
                                      label='Health Issues')
    transportation = forms.ChoiceField(
        choices=[('Own vehicle', 'Own vehicle'), ('Public transport', 'Public transport')], label='Transportation')
    event_day = forms.IntegerField(label='Event Day')
    event_month = forms.IntegerField(label='Event Month')
    event_year = forms.IntegerField(label='Event Year')
    event_minutes = forms.IntegerField(label='Event Minutes')
    prior_commitments = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')], label='Prior Commitments')


def predict_attendance(request):
    if request.method == 'POST':
        form = AttendanceForm(request.POST)
        if form.is_valid():
            # Convert form data to DataFrame
            data = pd.DataFrame([form.cleaned_data])

            # Ensure the order of columns is the same as used during training
            required_columns = [
                'age', 'number_of_children', 'length_of_service', 'work_hours', 'overtime_days',
                'vacation_days', 'work_days_last_month', 'absent_days_last_month', 'event_interest',
                'event_day', 'event_month', 'event_year', 'event_minutes', 'gender_F', 'marital_status_Married',
                'department_HR', 'department_IT', 'department_Marketing', 'department_Sales', 'position_HR Manager',
                'position_Marketing Specialist', 'position_Sales Executive', 'employment_status_Part-time',
                'employment_status_Contract', 'shift_timing_Night', 'past_event_attendance_Yes',
                'event_type_preference_Professional', 'event_type_preference_Social',
                'event_timing_preference_Weekend', 'event_timing_preference_Evening', 'family_commitments_Yes',
                'health_issues_Mild', 'health_issues_Severe', 'transportation_Public transport',
                'event_location_Off-site', 'prior_commitments_Yes'
            ]
            data = pd.get_dummies(data).reindex(columns=required_columns, fill_value=0)

            # Normalize numerical features
            numerical_columns = [
                'age', 'number_of_children', 'length_of_service', 'work_hours', 'overtime_days',
                'vacation_days', 'work_days_last_month', 'absent_days_last_month', 'event_interest',
                'event_day', 'event_month', 'event_year', 'event_minutes'
            ]
            scaler = StandardScaler()
            data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

            # Make prediction
            prediction = model.predict(data)
            result = 'Will Attend' if prediction[0] == 1 else 'Will Not Attend'

            return render(request, 'attendance/result.html', {'result': result})

    else:
        form = AttendanceForm()

    return render(request, 'attendance/form.html', {'form': form})
