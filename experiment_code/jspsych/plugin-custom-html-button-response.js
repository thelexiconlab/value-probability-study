var jsPsychConnector = (function (jspsych) {
  'use strict';

  const info = {
      name: "connector",
      parameters: {
          /** The HTML string to be displayed */
          stimulus: {
              type: jspsych.ParameterType.HTML_STRING,
              pretty_name: "Stimulus",
              default: undefined,
          },
          /** Array containing the label(s) for the button(s). */
          choices: {
              type: jspsych.ParameterType.STRING,
              pretty_name: "Choices",
              default: undefined,
              array: true,
          },
          /** The HTML for creating button. Can create own style. Use the "%choice%" string to indicate where the label from the choices parameter should be inserted. */
          button_html: {
              type: jspsych.ParameterType.HTML_STRING,
              pretty_name: "Button HTML",
              default: '<button class="jspsych-btn">%choice%</button>',
              array: true,
          },
          /** Any content here will be displayed under the button(s). */
          prompt: {
              type: jspsych.ParameterType.HTML_STRING,
              pretty_name: "Prompt",
              default: null,
          },
          /** How long to show the stimulus. */
          stimulus_duration: {
              type: jspsych.ParameterType.INT,
              pretty_name: "Stimulus duration",
              default: null,
          },
          /** How long to show the trial. */
          trial_duration: {
              type: jspsych.ParameterType.INT,
              pretty_name: "Trial duration",
              default: null,
          },
          /** The vertical margin of the button. */
          margin_vertical: {
              type: jspsych.ParameterType.STRING,
              pretty_name: "Margin vertical",
              default: "8px",
          },
          /** The horizontal margin of the button. */
          margin_horizontal: {
              type: jspsych.ParameterType.STRING,
              pretty_name: "Margin horizontal",
              default: "8px",
          },
          /** If true, then trial will end when user responds. */
          response_ends_trial: {
              type: jspsych.ParameterType.BOOL,
              pretty_name: "Response ends trial",
              default: true,
          },
      },
  };
  /**
   * html-button-response
   * jsPsych plugin for displaying a stimulus and getting a button response
   * @author Josh de Leeuw
   * adapted by Thomas Mazzuchi
   * @see {@link https://www.jspsych.org/plugins/jspsych-html-button-response/ html-button-response plugin documentation on jspsych.org}
   */
  class ConnectorPlugin {
      constructor(jsPsych) {
          this.jsPsych = jsPsych;
      }
      trial(display_element, trial) {
          // display stimulus
          var html = '<div id="jspsych-html-button-response-stimulus">' + trial.stimulus + "</div>";
          //display buttons
          var buttons = [];
          if (Array.isArray(trial.button_html)) {
              if (trial.button_html.length == trial.choices.length) {
                  buttons = trial.button_html;
              }
              else {
                  console.error("Error in html-button-response plugin. The length of the button_html array does not equal the length of the choices array");
              }
          }
          else {
              for (var i = 0; i < trial.choices.length; i++) {
                  buttons.push(trial.button_html);
              }
          }
          html += '<div id="jspsych-html-button-response-btngroup">';
          for (var i = 0; i < trial.choices.length; i++) {
              var str = buttons[i].replace(/%choice%/g, trial.choices[i]);
              html +=
                  '<div class="jspsych-html-button-response-button" style="display: inline-block; border: 2px solid white; margin:' +
                      trial.margin_vertical +
                      " " +
                      trial.margin_horizontal +
                      '" id="jspsych-html-button-response-button-' +
                      i +
                      '" data-choice="' +
                      i +
                      '">' +
                      str +
                      "</div>";
          }
          html += "</div>";
          //show prompt if there is one
          if (trial.prompt !== null) {
              html += "<div>" + trial.prompt + "</div>";
          }

          //display "finish" button
          html += '<div class="finish_button" id="finish_button" style="display: inline-block; border: 1px solid black; padding 0px"> <button class="jspsych-btn" disabled="disabled">Finish</button> </div>'

          display_element.innerHTML = html;
          
          //set up array to keep track of button clicks
          var buttons_clicked = [];
          for (var i = 0; i < trial.choices.length; i++) {
            buttons_clicked.push(0);
          }

          // start time
          var start_time = performance.now();
          // add event listeners to buttons
          for (var i = 0; i < trial.choices.length; i++) {
              display_element
                  .querySelector("#jspsych-html-button-response-button-" + i)
                  .addEventListener("click", (e) => {
                  var btn_el = e.currentTarget;
                  var choice = btn_el.getAttribute("data-choice"); // don't use dataset for jsdom compatibility
                  after_response(choice);
              });
          }
          display_element
                  .querySelector("#finish_button")
                  .addEventListener("click", (e) => {
                  attempt_finish();
              });

          // store response
          var response = {
              rt: null,
              button: null,
          };

          // function to end trial when it is time
          const end_trial = () => {
              // kill any remaining setTimeout handlers
              this.jsPsych.pluginAPI.clearAllTimeouts();
              // gather the data to store for the trial
              var responses = [];
              for (var i = 0; i < trial.choices.length; i++) {
                if(buttons_clicked[i] == 1){
                    responses.push(trial.choices[i])
                }
              }
              
              var trial_data = {
                  rt: response.rt,
                  stimulus: trial.stimulus,
                  response: responses,
              };
              // clear the display
              display_element.innerHTML = "";
              // move on to the next trial
              this.jsPsych.finishTrial(trial_data);
          };

          // function to handle responses by the subject
          function after_response(choice) {
            //update button selection
            buttons_clicked[choice] = buttons_clicked[choice]*(-1)+1;
            console.log(buttons_clicked);

            //update the clicked button
            
            var btns = document.querySelectorAll(".jspsych-html-button-response-button");
            if (buttons_clicked[choice] == 1){
                btns[choice].style["border"] = "2px solid green";
            }
            else {
                btns[choice].style["border"] = "2px solid white";
            }

            //update finish button 
            var finish_button = document.querySelectorAll(".finish_button button");
            console.log(finish_button)
            if(buttons_clicked.reduce((a,b) => {return a+b}) == 2){
                finish_button[0].removeAttribute("disabled");
            }
            else {
                finish_button[0].setAttribute("disabled", "disabled");
            }
          }

          //function to check if exactly two buttons are pressed
          function attempt_finish() {
            if(buttons_clicked.reduce((a,b) => {return a+b}) == 2){
                //disable the buttons
                var btns = document.querySelectorAll(".jspsych-html-button-response-button button");
                for (var i = 0; i < btns.length; i++) {
                  //btns[i].removeEventListener('click');
                  btns[i].setAttribute("disabled", "disabled");
              }
                //measure rt
                var end_time = performance.now();
                var rt = Math.round(end_time - start_time);
                response.rt = rt;

                //end the trial
                end_trial();
            }
            else {
                // below not working, unsure why 
                // html += "<div id:'invalid_response_text'> Please select exactly two words from the board </div>"
                // display_element.innerHTML = html;
            }
          }

          // hide image if timing is set
          if (trial.stimulus_duration !== null) {
              this.jsPsych.pluginAPI.setTimeout(() => {
                  display_element.querySelector("#jspsych-html-button-response-stimulus").style.visibility = "hidden";
              }, trial.stimulus_duration);
          }
          // end trial if time limit is set
          if (trial.trial_duration !== null) {
              this.jsPsych.pluginAPI.setTimeout(end_trial, trial.trial_duration);
          }
      }
      simulate(trial, simulation_mode, simulation_options, load_callback) {
          if (simulation_mode == "data-only") {
              load_callback();
              this.simulate_data_only(trial, simulation_options);
          }
          if (simulation_mode == "visual") {
              this.simulate_visual(trial, simulation_options, load_callback);
          }
      }
      create_simulation_data(trial, simulation_options) {
          const default_data = {
              stimulus: trial.stimulus,
              rt: this.jsPsych.randomization.sampleExGaussian(500, 50, 1 / 150, true),
              response: this.jsPsych.randomization.randomInt(0, trial.choices.length - 1),
          };
          const data = this.jsPsych.pluginAPI.mergeSimulationData(default_data, simulation_options);
          this.jsPsych.pluginAPI.ensureSimulationDataConsistency(trial, data);
          return data;
      }
      simulate_data_only(trial, simulation_options) {
          const data = this.create_simulation_data(trial, simulation_options);
          this.jsPsych.finishTrial(data);
      }
      simulate_visual(trial, simulation_options, load_callback) {
          const data = this.create_simulation_data(trial, simulation_options);
          const display_element = this.jsPsych.getDisplayElement();
          this.trial(display_element, trial);
          load_callback();
          if (data.rt !== null) {
              this.jsPsych.pluginAPI.clickTarget(display_element.querySelector(`div[data-choice="${data.response}"] button`), data.rt);
          }
      }
  }
  ConnectorPlugin.info = info;

  return ConnectorPlugin;

})(jsPsychModule);
